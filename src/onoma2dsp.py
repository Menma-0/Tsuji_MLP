"""
オノマトペから音声処理までのエンドツーエンドCLIツール
"""
import argparse
import torch
import pickle
import numpy as np
import sys
import os
from pathlib import Path

# 親ディレクトリのモジュールをインポート
sys.path.append(os.path.dirname(__file__))

from preprocessing.katakana_to_phoneme import KatakanaToPhoneme
from preprocessing.phoneme_to_mora import PhonemeToMora
from preprocessing.feature_extractor import OnomatopoeiaFeatureExtractor
from models.mlp_model import Onoma2DSPMLP, DSPParameterMapping
from dsp.dsp_engine import DSPEngine
from utils.create_rwcp_dataset import create_dsp_template


class Onoma2DSP:
    """オノマトペから音声処理までのエンドツーエンドクラス"""

    def __init__(self, model_path: str, scaler_path: str = None,
                 device: str = 'cpu', sample_rate: int = 44100,
                 amplification_factor: float = 1.0,
                 lambda_att: float = 0.5):
        """
        Args:
            model_path: 学習済みモデルのパス
            scaler_path: スケーラーのパス
            device: デバイス ('cpu' or 'cuda')
            sample_rate: サンプリングレート
            amplification_factor: モデル出力の増幅率（デフォルト1.0）
            lambda_att: attention補正の強度（デフォルト0.5、0で無効）
        """
        self.device = device
        self.sample_rate = sample_rate
        self.amplification_factor = amplification_factor
        self.lambda_att = lambda_att

        # 前処理モジュール
        self.katakana_converter = KatakanaToPhoneme()
        self.mora_converter = PhonemeToMora()
        self.feature_extractor = OnomatopoeiaFeatureExtractor()

        # モデルをロード
        self.model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=32, use_tanh=True)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # スケーラーをロード
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None
            print("Warning: Scaler not found. Using raw features.")

        # DSPエンジン
        self.dsp_engine = DSPEngine(sample_rate=sample_rate)
        self.mapper = DSPParameterMapping()

    def process(self, source_onomatopoeia: str, target_onomatopoeia: str,
               input_audio_path: str, output_audio_path: str, verbose: bool = True) -> dict:
        """
        2つのオノマトペの差分から音声処理まで実行

        Args:
            source_onomatopoeia: 現在の音声を表すオノマトペ（カタカナ）
            target_onomatopoeia: 編集後の音声を表すオノマトペ（カタカナ）
            input_audio_path: 入力音声ファイルパス
            output_audio_path: 出力音声ファイルパス
            verbose: 詳細表示するか

        Returns:
            処理結果の辞書
        """
        if verbose:
            print("\n" + "="*60)
            print(f"DIFFERENTIAL ONOMATOPOEIA TO DSP PROCESSING")
            print("="*60)
            print(f"\nSource (current sound): {source_onomatopoeia}")
            print(f"Target (desired sound): {target_onomatopoeia}")
            print(f"Input audio: {input_audio_path}")
            print(f"Output audio: {output_audio_path}")
            print()

        # ステップ1: 2つのオノマトペから特徴量抽出
        if verbose:
            print("[1/4] Extracting features from source onomatopoeia...")

        # Source特徴量
        source_phonemes = self.katakana_converter.convert(source_onomatopoeia)
        source_moras = self.mora_converter.convert(source_phonemes)
        source_features = self.feature_extractor.extract_features(source_phonemes, source_moras)

        if verbose:
            print(f"  Source phonemes: {source_phonemes}")
            print(f"  Source moras: {[''.join(m) for m in source_moras]}")

        # Target特徴量
        if verbose:
            print("\n[2/4] Extracting features from target onomatopoeia...")

        target_phonemes = self.katakana_converter.convert(target_onomatopoeia)
        target_moras = self.mora_converter.convert(target_phonemes)
        target_features = self.feature_extractor.extract_features(target_phonemes, target_moras)

        if verbose:
            print(f"  Target phonemes: {target_phonemes}")
            print(f"  Target moras: {[''.join(m) for m in target_moras]}")

        # 差分を計算
        feature_diff = target_features - source_features

        if self.scaler is not None:
            # 差分もスケーリング（元の特徴量と同じスケールで）
            feature_diff = self.scaler.transform(feature_diff.reshape(1, -1))[0]

        if verbose:
            print(f"\n  Feature difference vector: 38 dimensions")
            print(f"  Difference magnitude: {np.linalg.norm(feature_diff):.3f}")

        # ステップ3: 差分からDSPパラメータを推論
        if verbose:
            print("\n[3/4] Predicting DSP parameters from feature difference...")

        with torch.no_grad():
            diff_tensor = torch.FloatTensor(feature_diff).unsqueeze(0).to(self.device)
            normalized_params = self.model(diff_tensor).cpu().numpy()[0]

        # モデル出力を増幅（より大きな変化を生み出すため）
        normalized_params = np.clip(normalized_params * self.amplification_factor, -1.0, 1.0)

        # Attention機能: source_onomaから「ユーザがどこを聞いているか」を推定
        if self.lambda_att > 0:
            # source_onomaのDSPテンプレートを取得
            template_source = create_dsp_template(source_onomatopoeia)

            # 10次元配列に変換
            temp_array = np.array(template_source)

            # 絶対値を取って注目度ベクトルにする
            attention = np.abs(temp_array)

            # 0-1に正規化
            max_att = np.max(attention)
            if max_att > 1e-8:
                attention = attention / max_att
            attention = np.clip(attention, 0.0, 1.0)

            # normalized_paramsを補正
            # attention が大きい次元ほど変化を強調
            normalized_params = normalized_params * (1.0 + self.lambda_att * attention)

            # 再度クリッピング
            normalized_params = np.clip(normalized_params, -1.0, 1.0)

            if verbose:
                print(f"\n  [Attention Correction]")
                print(f"    lambda_att: {self.lambda_att:.2f}")
                print(f"    Attention vector (top 5):")
                # 注目度が高い順にソート
                att_sorted = sorted(enumerate(attention), key=lambda x: x[1], reverse=True)
                param_names = ['gain', 'comp', 'eq_sub', 'eq_low', 'eq_mid',
                             'eq_high', 'eq_pres', 'atk', 'sus', 'stretch']
                for idx, val in att_sorted[:5]:
                    print(f"      {param_names[idx]:<10}: {val:.3f}")

        mapped_params = self.mapper.map_parameters(normalized_params)

        if verbose:
            print("  Predicted parameters:")
            for key, value in mapped_params.items():
                if 'db' in key:
                    print(f"    {key:<25}: {value:>7.2f} dB")
                elif 'ratio' in key:
                    print(f"    {key:<25}: {value:>7.2f}x")
                else:
                    print(f"    {key:<25}: {value:>7.2f}")

        # ステップ4: 音声処理
        if verbose:
            print(f"\n[4/4] Applying DSP effects to audio...")

        self.dsp_engine.process_audio_file(
            input_audio_path,
            output_audio_path,
            mapped_params
        )

        if verbose:
            print("="*60)
            print("PROCESSING COMPLETED SUCCESSFULLY!")
            print("="*60 + "\n")

        result = {
            'source_onomatopoeia': source_onomatopoeia,
            'target_onomatopoeia': target_onomatopoeia,
            'source_phonemes': source_phonemes,
            'target_phonemes': target_phonemes,
            'source_moras': [''.join(m) for m in source_moras],
            'target_moras': [''.join(m) for m in target_moras],
            'feature_diff_magnitude': float(np.linalg.norm(feature_diff)),
            'normalized_params': normalized_params.tolist(),
            'mapped_params': mapped_params,
            'input_audio': input_audio_path,
            'output_audio': output_audio_path
        }

        return result

    def predict_dsp_params(self, source_onomatopoeia: str, target_onomatopoeia: str,
                          verbose: bool = False) -> dict:
        """
        オノマトペからDSPパラメータのみを予測（音声処理なし）

        Args:
            source_onomatopoeia: 現在の音声を表すオノマトペ（カタカナ）
            target_onomatopoeia: 編集後の音声を表すオノマトペ（カタカナ）
            verbose: 詳細表示するか

        Returns:
            DSPパラメータの辞書
        """
        if verbose:
            print(f"\n[Predicting DSP Parameters]")
            print(f"  Source: {source_onomatopoeia}")
            print(f"  Target: {target_onomatopoeia}")

        # Source特徴量
        source_phonemes = self.katakana_converter.convert(source_onomatopoeia)
        source_moras = self.mora_converter.convert(source_phonemes)
        source_features = self.feature_extractor.extract_features(source_phonemes, source_moras)

        # Target特徴量
        target_phonemes = self.katakana_converter.convert(target_onomatopoeia)
        target_moras = self.mora_converter.convert(target_phonemes)
        target_features = self.feature_extractor.extract_features(target_phonemes, target_moras)

        # 差分を計算
        feature_diff = target_features - source_features

        if self.scaler is not None:
            feature_diff = self.scaler.transform(feature_diff.reshape(1, -1))[0]

        # モデルで推論
        with torch.no_grad():
            diff_tensor = torch.FloatTensor(feature_diff).unsqueeze(0).to(self.device)
            normalized_params = self.model(diff_tensor).cpu().numpy()[0]

        # 増幅
        normalized_params = np.clip(normalized_params * self.amplification_factor, -1.0, 1.0)

        # Attention補正
        if self.lambda_att > 0:
            template_source = create_dsp_template(source_onomatopoeia)
            temp_array = np.array(template_source)
            attention = np.abs(temp_array)
            max_att = np.max(attention)
            if max_att > 1e-8:
                attention = attention / max_att
            attention = np.clip(attention, 0.0, 1.0)
            normalized_params = normalized_params * (1.0 + self.lambda_att * attention)
            normalized_params = np.clip(normalized_params, -1.0, 1.0)

        # パラメータマッピング
        mapped_params = self.mapper.map_parameters(normalized_params)

        if verbose:
            print(f"  Predicted parameters:")
            for key, value in mapped_params.items():
                print(f"    {key:<25}: {value:>7.2f}")

        return mapped_params

    def apply_dsp_only(self, input_audio_path: str, output_audio_path: str,
                      dsp_params: dict, verbose: bool = False):
        """
        与えられたDSPパラメータで音声処理のみ実行

        Args:
            input_audio_path: 入力音声ファイルパス
            output_audio_path: 出力音声ファイルパス
            dsp_params: DSPパラメータの辞書
            verbose: 詳細表示するか
        """
        if verbose:
            print(f"\n[Applying DSP Effects]")
            print(f"  Input: {input_audio_path}")
            print(f"  Output: {output_audio_path}")
            print(f"  Parameters:")
            for key, value in dsp_params.items():
                print(f"    {key:<25}: {value:>7.2f}")

        self.dsp_engine.process_audio_file(
            input_audio_path,
            output_audio_path,
            dsp_params
        )

        if verbose:
            print(f"  Processing completed!")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='2つのオノマトペの差分から音声にDSPエフェクトを適用'
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='現在の音声を表すオノマトペ（カタカナ）'
    )
    parser.add_argument(
        '--target', '-t',
        type=str,
        required=True,
        help='編集後の音声を表すオノマトペ（カタカナ）'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='入力音声ファイルパス'
    )
    parser.add_argument(
        '--output', '-p',
        type=str,
        required=True,
        help='出力音声ファイルパス'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/saved_model.pth',
        help='学習済みモデルのパス'
    )
    parser.add_argument(
        '--scaler', '-c',
        type=str,
        default='models/scaler.pkl',
        help='スケーラーのパス'
    )
    parser.add_argument(
        '--sample-rate', '-sr',
        type=int,
        default=44100,
        help='サンプリングレート'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='デバイス'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='詳細表示をオフにする'
    )

    args = parser.parse_args()

    # ファイルの存在確認
    if not os.path.exists(args.input):
        print(f"Error: Input audio file not found: {args.input}")
        return

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first using train.py")
        return

    # 処理実行
    try:
        processor = Onoma2DSP(
            model_path=args.model,
            scaler_path=args.scaler,
            device=args.device,
            sample_rate=args.sample_rate
        )

        result = processor.process(
            args.source,
            args.target,
            args.input,
            args.output,
            verbose=not args.quiet
        )

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
