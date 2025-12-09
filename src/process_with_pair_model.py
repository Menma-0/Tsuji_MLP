"""
新しいペアモデルを使用した音声編集処理

入力: source_onomatopoeia, target_onomatopoeia, input_audio
出力: 編集された音声ファイル
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
from src.models.mlp_model import Onoma2DSPMLP, DSPParameterMapping
from src.dsp.dsp_engine import DSPEngine


class PairModelProcessor:
    """ペアモデルを使用した音声処理クラス"""

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        sample_rate: int = 44100,
        device: str = 'cpu'
    ):
        """
        Args:
            model_path: 学習済みモデルのパス (pair_model.pth)
            scaler_path: スケーラーのパス (pair_scaler.pkl)
            sample_rate: サンプリングレート
            device: デバイス
        """
        self.device = device
        self.sample_rate = sample_rate

        # 前処理モジュール
        self.katakana_converter = KatakanaToPhoneme()
        self.mora_converter = PhonemeToMora()
        self.feature_extractor = OnomatopoeiaFeatureExtractor()

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

    def _extract_features(self, onomatopoeia: str) -> np.ndarray:
        """オノマトペから特徴量を抽出"""
        phonemes = self.katakana_converter.convert(onomatopoeia)
        moras = self.mora_converter.convert(phonemes)
        features = self.feature_extractor.extract_features(phonemes, moras)
        return features, phonemes, moras

    def process(
        self,
        source_onomatopoeia: str,
        target_onomatopoeia: str,
        input_audio_path: str,
        output_audio_path: str,
        verbose: bool = True
    ) -> dict:
        """
        音声編集を実行

        Args:
            source_onomatopoeia: 元の音を表すオノマトペ
            target_onomatopoeia: 目標の音を表すオノマトペ
            input_audio_path: 入力音声パス
            output_audio_path: 出力音声パス
            verbose: 詳細表示
        """
        if verbose:
            print("=" * 70)
            print("PAIR MODEL AUDIO PROCESSING")
            print("=" * 70)
            print(f"\nSource onomatopoeia: {source_onomatopoeia}")
            print(f"Target onomatopoeia: {target_onomatopoeia}")
            print(f"Input audio: {input_audio_path}")
            print(f"Output audio: {output_audio_path}")

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

        # DSP差分を実際のパラメータにマッピング
        # 差分なので-2〜+2の範囲がありえる、-1〜+1にクリップしてからマッピング
        dsp_diff_clipped = np.clip(dsp_diff, -1.0, 1.0)
        mapped_params = self.mapper.map_parameters(dsp_diff_clipped)

        if verbose:
            param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                          'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']
            print("\n  Raw DSP differences (model output):")
            for i, name in enumerate(param_names):
                print(f"    {name:<15}: {dsp_diff[i]:>+8.4f}")

            print("\n  Mapped DSP parameters:")
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

        self.dsp_engine.process_audio_file(
            input_audio_path,
            output_audio_path,
            mapped_params
        )

        if verbose:
            print("\n" + "=" * 70)
            print("PROCESSING COMPLETED!")
            print("=" * 70)
            print(f"\nOutput saved to: {output_audio_path}")

        return {
            'source_onomatopoeia': source_onomatopoeia,
            'target_onomatopoeia': target_onomatopoeia,
            'source_phonemes': source_phonemes,
            'target_phonemes': target_phonemes,
            'source_moras': [''.join(m) for m in source_moras],
            'target_moras': [''.join(m) for m in target_moras],
            'feature_diff_magnitude': float(np.linalg.norm(feature_diff)),
            'dsp_diff_raw': dsp_diff.tolist(),
            'mapped_params': mapped_params,
            'input_audio': input_audio_path,
            'output_audio': output_audio_path
        }


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
