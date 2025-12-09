"""
新しいペアモデルを使用した音声編集処理

入力: source_onomatopoeia, target_onomatopoeia, input_audio
出力: 編集された音声ファイル

改善点:
- 累積DSPパラメータ管理：複数回の編集でも元音から再レンダリング
- 減速ロジック：極端なパラメータへの変化を制限
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


class PairModelProcessor:
    """ペアモデルを使用した音声処理クラス"""

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        sample_rate: int = 44100,
        device: str = 'cpu',
        lambda_att: float = 0.5,
        use_cumulative: bool = True
    ):
        """
        Args:
            model_path: 学習済みモデルのパス (pair_model.pth)
            scaler_path: スケーラーのパス (pair_scaler.pkl)
            sample_rate: サンプリングレート
            device: デバイス
            lambda_att: Attention補正の強度 (0.0で無効、1.0で最大)
            use_cumulative: 累積DSP管理を使用するかどうか
        """
        self.device = device
        self.sample_rate = sample_rate
        self.lambda_att = lambda_att
        self.use_cumulative = use_cumulative

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
        self.cumulative_manager = CumulativeDSPManager(sample_rate=sample_rate) if use_cumulative else None

    def _extract_features(self, onomatopoeia: str) -> np.ndarray:
        """オノマトペから特徴量を抽出"""
        phonemes = self.katakana_converter.convert(onomatopoeia)
        moras = self.mora_converter.convert(phonemes)
        features = self.feature_extractor.extract_features(phonemes, moras)
        return features, phonemes, moras

    def start_new_session(self, input_audio_path: str):
        """
        新しい編集セッションを開始（元音を設定）

        Args:
            input_audio_path: 元音声のパス
        """
        if self.cumulative_manager:
            self.cumulative_manager.set_original_audio(input_audio_path, force_reset=True)

    def continue_session(self, input_audio_path: str = None) -> bool:
        """
        既存のセッションを継続

        Args:
            input_audio_path: 確認用のオーディオパス

        Returns:
            セッションを継続できたかどうか
        """
        if self.cumulative_manager:
            return self.cumulative_manager.continue_session(input_audio_path)
        return False

    def get_cumulative_state(self) -> dict:
        """累積状態を取得"""
        if self.cumulative_manager:
            return self.cumulative_manager.get_current_state()
        return None

    def reset_cumulative(self):
        """累積パラメータをリセット"""
        if self.cumulative_manager:
            self.cumulative_manager.reset()

    def undo_last_edit(self) -> bool:
        """最後の編集を取り消す"""
        if self.cumulative_manager:
            return self.cumulative_manager.undo_last_edit()
        return False

    def process(
        self,
        source_onomatopoeia: str,
        target_onomatopoeia: str,
        input_audio_path: str,
        output_audio_path: str,
        verbose: bool = True,
        continue_session: bool = True
    ) -> dict:
        """
        音声編集を実行

        Args:
            source_onomatopoeia: 元の音を表すオノマトペ
            target_onomatopoeia: 目標の音を表すオノマトペ
            input_audio_path: 入力音声パス
            output_audio_path: 出力音声パス
            verbose: 詳細表示
            continue_session: 既存のセッションを継続するかどうか（False=常に新規セッション）
        """
        if verbose:
            print("=" * 70)
            print("PAIR MODEL AUDIO PROCESSING")
            if self.use_cumulative:
                print("(Cumulative Mode: ON)")
            print("=" * 70)
            print(f"\nSource onomatopoeia: {source_onomatopoeia}")
            print(f"Target onomatopoeia: {target_onomatopoeia}")
            print(f"Input audio: {input_audio_path}")
            print(f"Output audio: {output_audio_path}")

        # 累積モードの場合、セッション管理
        is_continuing = False
        if self.use_cumulative and self.cumulative_manager:
            if continue_session:
                is_continuing = self.cumulative_manager.continue_session(input_audio_path)

            if not is_continuing:
                self.cumulative_manager.set_original_audio(input_audio_path)

            if verbose and is_continuing:
                print(f"\n[Continuing session - Edit #{len(self.cumulative_manager.edit_history) + 1}]")
                print(f"  Current params: {self.cumulative_manager.get_cumulative_params_summary()}")

        # 1. 特徴量抽出
        if verbose:
            print("\n[1/4] Extracting features...")

        source_features, source_phonemes, source_moras = self._extract_features(source_onomatopoeia)
        target_features, target_phonemes, target_moras = self._extract_features(target_onomatopoeia)

        if verbose:
            print(f"  Source: {source_onomatopoeia}")
            print(f"    Phonemes: {source_phonemes}")
            print(f"    Moras: {[''.join(m) for m in source_moras]}")
            print(f"  Target: {target_onomatopoeia}")
            print(f"    Phonemes: {target_phonemes}")
            print(f"    Moras: {[''.join(m) for m in target_moras]}")

        # 2. 特徴量差分を計算
        if verbose:
            print("\n[2/4] Computing feature difference...")

        feature_diff = target_features - source_features

        # スケーリング
        feature_diff_scaled = self.scaler.transform(feature_diff.reshape(1, -1))[0]

        if verbose:
            print(f"  Feature diff magnitude (raw): {np.linalg.norm(feature_diff):.4f}")
            print(f"  Feature diff magnitude (scaled): {np.linalg.norm(feature_diff_scaled):.4f}")

        # 3. DSPパラメータ差分を予測
        if verbose:
            print("\n[3/4] Predicting DSP parameter differences...")

        with torch.no_grad():
            diff_tensor = torch.FloatTensor(feature_diff_scaled).unsqueeze(0).to(self.device)
            dsp_diff = self.model(diff_tensor).cpu().numpy()[0]

        # Attention補正を適用
        attention_info = None
        if self.lambda_att > 0:
            # sourceとtarget両方のAttentionを計算し、変化の方向を考慮
            source_attention = self.attention_calculator.compute_attention(source_phonemes, source_moras)
            target_attention = self.attention_calculator.compute_attention(target_phonemes, target_moras)

            # Attention差分: target側で重要な特徴を強調
            # target_attention が高い次元 → その方向への変化を強調
            # source_attention が高い次元 → 元の特徴からの脱却を強調
            attention_diff = target_attention - source_attention * 0.3
            attention_diff = np.clip(attention_diff, 0.0, 1.0)

            # 両方のAttentionを組み合わせた総合的な重み
            combined_attention = (target_attention * 0.7 + source_attention * 0.3)
            combined_attention = np.clip(combined_attention, 0.0, 1.0)

            # DSP差分にAttention補正を適用
            # Attentionが高い次元ほど変化を強調（1.0 + lambda * attention）
            dsp_diff_original = dsp_diff.copy()
            dsp_diff = dsp_diff * (1.0 + self.lambda_att * combined_attention)

            attention_info = {
                'source_attention': source_attention,
                'target_attention': target_attention,
                'combined_attention': combined_attention,
                'dsp_diff_before': dsp_diff_original,
                'dsp_diff_after': dsp_diff
            }

            if verbose:
                print(f"\n  [Attention Correction] (lambda={self.lambda_att:.2f})")
                param_names_short = ['gain', 'comp', 'sub', 'low', 'mid', 'high', 'pres', 'atk', 'sus', 'str']
                print(f"    {'Param':<8} {'Source':>8} {'Target':>8} {'Combined':>8} {'Before':>8} {'After':>8}")
                print("    " + "-" * 56)
                for i, name in enumerate(param_names_short):
                    print(f"    {name:<8} {source_attention[i]:>8.3f} {target_attention[i]:>8.3f} "
                          f"{combined_attention[i]:>8.3f} {dsp_diff_original[i]:>+8.3f} {dsp_diff[i]:>+8.3f}")

        # 累積モードの処理
        effective_dsp_diff = dsp_diff
        if self.use_cumulative and self.cumulative_manager:
            if verbose:
                print("\n  [Deceleration Logic]")

            # 減速ロジックを適用
            effective_dsp_diff = self.cumulative_manager.apply_deceleration(dsp_diff, verbose=verbose)

            # 累積パラメータを更新
            self.cumulative_manager.update_parameters(
                effective_dsp_diff,
                source_onoma=source_onomatopoeia,
                target_onoma=target_onomatopoeia
            )

            if verbose:
                print(f"\n  Cumulative params after update: {self.cumulative_manager.get_cumulative_params_summary()}")

        # DSP差分を実際のパラメータにマッピング
        if self.use_cumulative and self.cumulative_manager:
            # 累積モード: 累積パラメータをマッピング
            final_params = self.cumulative_manager.cumulative_params
            mapped_params = self.mapper.map_parameters(final_params)
        else:
            # 通常モード: 今回の差分のみをマッピング
            dsp_diff_clipped = np.clip(dsp_diff, -1.0, 1.0)
            mapped_params = self.mapper.map_parameters(dsp_diff_clipped)

        if verbose:
            param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                          'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']
            print("\n  Raw DSP differences (model output):")
            for i, name in enumerate(param_names):
                print(f"    {name:<15}: {dsp_diff[i]:>+8.4f}")

            if self.use_cumulative and self.cumulative_manager:
                print("\n  Effective DSP differences (after deceleration):")
                for i, name in enumerate(param_names):
                    print(f"    {name:<15}: {effective_dsp_diff[i]:>+8.4f}")

            print("\n  Mapped DSP parameters (final):")
            for key, value in mapped_params.items():
                if 'db' in key:
                    print(f"    {key:<25}: {value:>+8.2f} dB")
                elif 'ratio' in key:
                    print(f"    {key:<25}: {value:>8.2f}x")
                else:
                    print(f"    {key:<25}: {value:>+8.2f}")

        # 4. 音声処理を適用
        if verbose:
            print("\n[4/4] Applying DSP effects...")

        if self.use_cumulative and self.cumulative_manager:
            # 累積モード: 元音から直接レンダリング
            if verbose:
                print("  (Rendering from original audio with cumulative parameters)")
            self.cumulative_manager.render(output_audio_path, verbose=False)
        else:
            # 通常モード: 入力音声に適用
            self.dsp_engine.process_audio_file(
                input_audio_path,
                output_audio_path,
                mapped_params
            )

        if verbose:
            print("\n" + "=" * 70)
            print("PROCESSING COMPLETED!")
            if self.use_cumulative:
                edit_count = len(self.cumulative_manager.edit_history) if self.cumulative_manager else 0
                print(f"(Cumulative Mode - Total edits: {edit_count})")
            print("=" * 70)
            print(f"\nOutput saved to: {output_audio_path}")

        result = {
            'source_onomatopoeia': source_onomatopoeia,
            'target_onomatopoeia': target_onomatopoeia,
            'source_phonemes': source_phonemes,
            'target_phonemes': target_phonemes,
            'source_moras': [''.join(m) for m in source_moras],
            'target_moras': [''.join(m) for m in target_moras],
            'feature_diff_magnitude': float(np.linalg.norm(feature_diff)),
            'dsp_diff_raw': dsp_diff.tolist(),
            'dsp_diff_effective': effective_dsp_diff.tolist() if self.use_cumulative else dsp_diff.tolist(),
            'mapped_params': mapped_params,
            'input_audio': input_audio_path,
            'output_audio': output_audio_path,
            'lambda_att': self.lambda_att,
            'cumulative_mode': self.use_cumulative
        }

        # 累積モードの情報を追加
        if self.use_cumulative and self.cumulative_manager:
            result['cumulative'] = {
                'edit_number': len(self.cumulative_manager.edit_history),
                'cumulative_params': self.cumulative_manager.cumulative_params.tolist(),
                'is_continuing': is_continuing
            }

        # Attention情報を追加（有効な場合）
        if attention_info is not None:
            result['attention'] = {
                'source': attention_info['source_attention'].tolist(),
                'target': attention_info['target_attention'].tolist(),
                'combined': attention_info['combined_attention'].tolist()
            }

        return result


def main():
    """足音データでテスト"""
    print("=" * 70)
    print("足音データでのペアモデルテスト")
    print("=" * 70)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # パス設定
    model_path = os.path.join(project_root, 'models', 'pair_model.pth')
    scaler_path = os.path.join(project_root, 'models', 'pair_scaler.pkl')
    input_audio = os.path.join(project_root, 'demo_audio', 'test_walk.wav')
    output_dir = os.path.join(project_root, 'demo_audio', 'pair_model_test')

    os.makedirs(output_dir, exist_ok=True)

    # プロセッサを初期化
    processor = PairModelProcessor(
        model_path=model_path,
        scaler_path=scaler_path
    )

    # テストケース（前回と同じ: ジャッジャッ → タッタッ）
    test_cases = [
        ('ジャッジャッ', 'タッタッ', 'walk_jajja_to_tatta.wav'),
        ('ジャッジャッ', 'ドスドス', 'walk_jajja_to_dosudosu.wav'),
        ('ジャッジャッ', 'サクサク', 'walk_jajja_to_sakusaku.wav'),
        ('ジャッジャッ', 'ペタペタ', 'walk_jajja_to_petapeta.wav'),
    ]

    results = []

    for source, target, output_name in test_cases:
        print(f"\n\n{'='*70}")
        print(f"TEST: {source} → {target}")
        print(f"{'='*70}")

        output_path = os.path.join(output_dir, output_name)

        try:
            result = processor.process(
                source_onomatopoeia=source,
                target_onomatopoeia=target,
                input_audio_path=input_audio,
                output_audio_path=output_path,
                verbose=True
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # サマリー
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nProcessed {len(results)} test cases")
    print(f"Output directory: {output_dir}")

    print("\nGenerated files:")
    for result in results:
        print(f"  - {os.path.basename(result['output_audio'])}")

    print("\nDSP parameter comparison:")
    param_names = ['gain_db', 'eq_sub_db', 'eq_low_db', 'eq_mid_db',
                   'eq_high_db', 'eq_presence_db', 'transient_attack']

    print(f"\n{'Case':<25} ", end='')
    for name in param_names[:4]:
        print(f"{name:<12}", end='')
    print()
    print("-" * 75)

    for result in results:
        case_name = f"{result['source_onomatopoeia']}→{result['target_onomatopoeia']}"
        print(f"{case_name:<25} ", end='')
        for name in param_names[:4]:
            val = result['mapped_params'].get(name, 0)
            print(f"{val:>+10.2f}  ", end='')
        print()


if __name__ == '__main__':
    main()
