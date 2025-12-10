"""
累積DSPパラメータによる音声編集処理

方針A: 元音+累積パラメータから毎回再レンダリング
方針B: パラメータ上限・下限と減速ロジック
"""
import torch
import pickle
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from src.preprocessing.katakana_to_phoneme import KatakanaToPhoneme
from src.preprocessing.phoneme_to_mora import PhonemeToMora
from src.preprocessing.feature_extractor import OnomatopoeiaFeatureExtractor
from src.preprocessing.phoneme_attention import PhonemeAttention
from src.models.mlp_model import Onoma2DSPMLP, DSPParameterMapping
from src.dsp.dsp_engine import DSPEngine
from src.utils.cumulative_dsp_manager import CumulativeDSPManager


class CumulativeDSPProcessor:
    """累積DSPパラメータを使用した音声処理クラス（方針A&B対応）"""

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        sample_rate: int = 44100,
        device: str = 'cpu',
        lambda_att: float = 0.5
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.lambda_att = lambda_att

        # 前処理モジュール
        self.katakana_converter = KatakanaToPhoneme()
        self.mora_converter = PhonemeToMora()
        self.feature_extractor = OnomatopoeiaFeatureExtractor()
        self.attention_calculator = PhonemeAttention()

        # モデルをロード
        self.model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=64, use_tanh=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.to(device)
        self.model.eval()

        # スケーラーをロード
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # 累積DSPマネージャー
        self.cumulative_manager = CumulativeDSPManager(sample_rate=sample_rate)

    def start_session(self, original_audio_path: str) -> str:
        """セッションを開始"""
        self.cumulative_manager.set_original_audio(original_audio_path, force_reset=True)
        return self.cumulative_manager.original_audio_backup_path

    def _extract_features(self, onomatopoeia: str) -> tuple:
        phonemes = self.katakana_converter.convert(onomatopoeia)
        moras = self.mora_converter.convert(phonemes)
        features = self.feature_extractor.extract_features(phonemes, moras)
        return features, phonemes, moras

    def _predict_dsp_delta(
        self,
        source_onomatopoeia: str,
        target_onomatopoeia: str,
        verbose: bool = False
    ) -> np.ndarray:
        source_features, source_phonemes, source_moras = self._extract_features(source_onomatopoeia)
        target_features, target_phonemes, target_moras = self._extract_features(target_onomatopoeia)

        if verbose:
            print(f"  Source: {source_onomatopoeia}")
            print(f"    Phonemes: {source_phonemes}")
            print(f"  Target: {target_onomatopoeia}")
            print(f"    Phonemes: {target_phonemes}")

        feature_diff = target_features - source_features
        feature_diff_scaled = self.scaler.transform(feature_diff.reshape(1, -1))[0]

        with torch.no_grad():
            diff_tensor = torch.FloatTensor(feature_diff_scaled).unsqueeze(0).to(self.device)
            dsp_diff = self.model(diff_tensor).cpu().numpy()[0]

        if self.lambda_att > 0:
            source_attention = self.attention_calculator.compute_attention(source_phonemes, source_moras)
            target_attention = self.attention_calculator.compute_attention(target_phonemes, target_moras)
            combined_attention = (target_attention * 0.7 + source_attention * 0.3)
            combined_attention = np.clip(combined_attention, 0.0, 1.0)
            dsp_diff = dsp_diff * (1.0 + self.lambda_att * combined_attention)

        return dsp_diff

    def process(
        self,
        source_onomatopoeia: str,
        target_onomatopoeia: str,
        output_audio_path: str,
        verbose: bool = True
    ) -> dict:
        if self.cumulative_manager.original_audio_backup_path is None:
            raise RuntimeError("セッションが開始されていません。start_session()を先に呼び出してください。")

        edit_num = len(self.cumulative_manager.edit_history) + 1

        if verbose:
            print("=" * 70)
            print(f"CUMULATIVE DSP EDITING - Edit #{edit_num}")
            print("=" * 70)
            print(f"\nSource: {source_onomatopoeia} → Target: {target_onomatopoeia}")

        if verbose:
            print("\n[1/3] Predicting DSP delta...")

        dsp_delta = self._predict_dsp_delta(
            source_onomatopoeia, target_onomatopoeia, verbose=verbose
        )

        if verbose:
            print("\n  Model output (raw delta):")
            param_names = self.cumulative_manager.PARAM_NAMES
            for i, name in enumerate(param_names):
                print(f"    {name:<15}: {dsp_delta[i]:>+8.4f}")

        if verbose:
            print("\n[2/3] Applying deceleration logic (方針B)...")

        effective_delta = self.cumulative_manager.apply_deceleration(dsp_delta, verbose=verbose)

        # 累積パラメータを更新
        self.cumulative_manager.update_parameters(
            effective_delta,
            source_onoma=source_onomatopoeia,
            target_onoma=target_onomatopoeia
        )

        if verbose:
            print("\n[3/3] Re-rendering from original audio (方針A)...")

        cumulative_params = self.cumulative_manager.cumulative_params.copy()

        if verbose:
            print("\n  Cumulative DSP parameters:")
            param_names = self.cumulative_manager.PARAM_NAMES
            for i, name in enumerate(param_names):
                print(f"    {name:<15}: {cumulative_params[i]:>+8.4f}")

        # 元音声から累積パラメータで再レンダリング
        self.cumulative_manager.render(output_audio_path, verbose=verbose)

        if verbose:
            print("\n" + "=" * 70)
            print(f"EDIT #{edit_num} COMPLETED!")
            print("=" * 70)
            print(f"\nOriginal audio: {self.cumulative_manager.original_audio_backup_path}")
            print(f"Output saved to: {output_audio_path}")

        return {
            'edit_number': edit_num,
            'source_onomatopoeia': source_onomatopoeia,
            'target_onomatopoeia': target_onomatopoeia,
            'dsp_delta_raw': dsp_delta.tolist(),
            'dsp_delta_effective': effective_delta.tolist(),
            'cumulative_params': cumulative_params.tolist(),
            'original_audio': self.cumulative_manager.original_audio_backup_path,
            'output_audio': output_audio_path
        }

    def get_status(self) -> dict:
        return self.cumulative_manager.get_current_state()

    def reset_session(self):
        self.cumulative_manager.reset()
