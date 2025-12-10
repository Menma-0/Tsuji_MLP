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
from src.dsp.cumulative_dsp import CumulativeDSPManager, get_cumulative_manager


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
        """
        Args:
            model_path: 学習済みモデルのパス (pair_model.pth)
            scaler_path: スケーラーのパス (pair_scaler.pkl)
            sample_rate: サンプリングレート
            device: デバイス
            lambda_att: Attention補正の強度 (0.0で無効、1.0で最大)
        """
        self.device = device
        self.sample_rate = sample_rate
        self.lambda_att = lambda_att

        # 前処理モジュール
        self.katakana_converter = KatakanaToPhoneme()
        self.mora_converter = PhonemeToMora()
        self.feature_extractor = OnomatopoeiaFeatureExtractor()
        self.attention_calculator = PhonemeAttention()

        # モデルをロード（ペアモデルは hidden_dim=64, use_tanh=False）
        self.model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=64, use_tanh=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.to(device)
        self.model.eval()

        # スケーラーをロード
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # DSPエンジン
        self.dsp_engine = DSPEngine(sample_rate=sample_rate)
        self.mapper = DSPParameterMapping()

        # 累積DSPマネージャー
        self.cumulative_manager = CumulativeDSPManager()

    def start_session(self, original_audio_path: str) -> str:
        """
        新しい編集セッションを開始

        Args:
            original_audio_path: 元音声のパス

        Returns:
            バックアップされた元音声のパス
        """
        return self.cumulative_manager.start_session(original_audio_path)

    def _extract_features(self, onomatopoeia: str) -> tuple:
        """オノマトペから特徴量を抽出"""
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
        """
        オノマトペペアからDSPパラメータ差分を予測

        Returns:
            DSPパラメータ差分（10次元、正規化された値）
        """
        # 特徴量抽出
        source_features, source_phonemes, source_moras = self._extract_features(source_onomatopoeia)
        target_features, target_phonemes, target_moras = self._extract_features(target_onomatopoeia)

        if verbose:
            print(f"  Source: {source_onomatopoeia}")
            print(f"    Phonemes: {source_phonemes}")
            print(f"  Target: {target_onomatopoeia}")
            print(f"    Phonemes: {target_phonemes}")

        # 特徴量差分を計算
        feature_diff = target_features - source_features
        feature_diff_scaled = self.scaler.transform(feature_diff.reshape(1, -1))[0]

        # DSPパラメータ差分を予測
        with torch.no_grad():
            diff_tensor = torch.FloatTensor(feature_diff_scaled).unsqueeze(0).to(self.device)
            dsp_diff = self.model(diff_tensor).cpu().numpy()[0]

        # Attention補正を適用
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
        """
        音声編集を実行（方針A&B適用）

        方針A: 元音声から累積パラメータで再レンダリング
        方針B: 減速ロジックを適用

        Args:
            source_onomatopoeia: 元の音を表すオノマトペ
            target_onomatopoeia: 目標の音を表すオノマトペ
            output_audio_path: 出力音声パス
            verbose: 詳細表示
        """
        if self.cumulative_manager.backup_audio_path is None:
            raise RuntimeError("セッションが開始されていません。start_session()を先に呼び出してください。")

        edit_num = self.cumulative_manager.edit_count + 1

        if verbose:
            print("=" * 70)
            print(f"CUMULATIVE DSP EDITING - Edit #{edit_num}")
            print("=" * 70)
            print(f"\nSource: {source_onomatopoeia} → Target: {target_onomatopoeia}")

        # 1. DSPパラメータ差分を予測
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

        # 2. 方針B: 減速ロジックを適用して累積パラメータを更新
        if verbose:
            print("\n[2/3] Applying deceleration logic (方針B)...")

        effective_delta = self.cumulative_manager.apply_delta_with_deceleration(
            dsp_delta, verbose=verbose
        )

        # 3. 方針A: 元音声から累積パラメータで再レンダリング
        if verbose:
            print("\n[3/3] Re-rendering from original audio (方針A)...")

        cumulative_params = self.cumulative_manager.get_cumulative_params()

        # 累積パラメータを実際のDSP値にマッピング
        cumulative_params_clipped = np.clip(cumulative_params, -1.0, 1.0)
        mapped_params = self.mapper.map_parameters(cumulative_params_clipped)

        if verbose:
            print("\n  Cumulative DSP parameters:")
            param_names = self.cumulative_manager.PARAM_NAMES
            for i, name in enumerate(param_names):
                print(f"    {name:<15}: {cumulative_params[i]:>+8.4f}")

            print("\n  Mapped DSP parameters:")
            for key, value in mapped_params.items():
                if 'db' in key:
                    print(f"    {key:<25}: {value:>+8.2f} dB")
                elif 'ratio' in key:
                    print(f"    {key:<25}: {value:>8.2f}x")
                else:
                    print(f"    {key:<25}: {value:>+8.2f}")

        # 元音声から再レンダリング
        original_path = self.cumulative_manager.backup_audio_path
        self.dsp_engine.process_audio_file(
            original_path,
            output_audio_path,
            mapped_params
        )

        if verbose:
            print("\n" + "=" * 70)
            print(f"EDIT #{edit_num} COMPLETED!")
            print("=" * 70)
            print(f"\nOriginal audio: {original_path}")
            print(f"Output saved to: {output_audio_path}")

        return {
            'edit_number': edit_num,
            'source_onomatopoeia': source_onomatopoeia,
            'target_onomatopoeia': target_onomatopoeia,
            'dsp_delta_raw': dsp_delta.tolist(),
            'dsp_delta_effective': effective_delta.tolist(),
            'cumulative_params': cumulative_params.tolist(),
            'mapped_params': mapped_params,
            'original_audio': original_path,
            'output_audio': output_audio_path
        }

    def get_status(self) -> dict:
        """現在の状態を取得"""
        return self.cumulative_manager.get_status()

    def reset_session(self):
        """セッションをリセット"""
        self.cumulative_manager.reset()


def main():
    """累積DSP編集のテスト（方針A&B）"""
    print("=" * 70)
    print("累積DSP編集テスト（方針A&B）")
    print("=" * 70)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # パス設定
    model_path = os.path.join(project_root, 'models', 'pair_model.pth')
    scaler_path = os.path.join(project_root, 'models', 'pair_scaler.pkl')
    input_audio = os.path.join(project_root, 'demo_audio', 'diverse_test', '02_coin_original.wav')
    output_dir = os.path.join(project_root, 'demo_audio', 'cumulative_test')

    os.makedirs(output_dir, exist_ok=True)

    # プロセッサを初期化
    processor = CumulativeDSPProcessor(
        model_path=model_path,
        scaler_path=scaler_path
    )

    # セッション開始
    print(f"\n入力音声: {input_audio}")
    backup_path = processor.start_session(input_audio)
    print(f"バックアップ: {backup_path}")

    # 同じ編集を3回繰り返す（チャリン → ガチャン）
    source = 'チャリン'
    target = 'ガチャン'

    print(f"\n\n{'='*70}")
    print(f"テスト: {source} → {target} を3回繰り返し")
    print(f"{'='*70}")

    results = []

    for i in range(3):
        output_path = os.path.join(output_dir, f'coin_cumulative_{i+1}x.wav')

        result = processor.process(
            source_onomatopoeia=source,
            target_onomatopoeia=target,
            output_audio_path=output_path,
            verbose=True
        )
        results.append(result)

        print("\n" + "-" * 70 + "\n")

    # サマリー
    print("\n" + "=" * 70)
    print("SUMMARY - 累積パラメータの推移")
    print("=" * 70)

    param_names = CumulativeDSPManager.PARAM_NAMES
    header = f"{'Edit':<6}"
    for name in param_names[:7]:  # 最初の7パラメータ
        header += f"{name:<10}"
    print(header)
    print("-" * 76)

    for result in results:
        row = f"#{result['edit_number']:<5}"
        for i in range(7):
            row += f"{result['cumulative_params'][i]:>+9.3f} "
        print(row)

    print("\n" + "=" * 70)
    print("方針A&Bの効果:")
    print("- 方針A: 毎回元音声から再レンダリング → フィルタの重ね掛け防止")
    print("- 方針B: 減速ロジック → パラメータが上限に近づくほど変化が緩やか")
    print("=" * 70)

    # 比較用: 従来方式（重ね掛け）も生成
    print("\n\n" + "=" * 70)
    print("比較: 従来方式（フィルタ重ね掛け）")
    print("=" * 70)

    from src.process_with_pair_model import PairModelProcessor

    old_processor = PairModelProcessor(
        model_path=model_path,
        scaler_path=scaler_path
    )

    # 従来方式: 3回重ね掛け
    current_input = input_audio
    for i in range(3):
        output_path = os.path.join(output_dir, f'coin_stacked_{i+1}x.wav')
        old_processor.process(
            source_onomatopoeia=source,
            target_onomatopoeia=target,
            input_audio_path=current_input,
            output_audio_path=output_path,
            verbose=False
        )
        current_input = output_path
        print(f"  Stacked {i+1}x: {output_path}")

    print("\n比較ファイル生成完了:")
    print("  cumulative_*x.wav: 方針A&B（累積パラメータ）")
    print("  stacked_*x.wav: 従来方式（フィルタ重ね掛け）")


if __name__ == '__main__':
    main()
