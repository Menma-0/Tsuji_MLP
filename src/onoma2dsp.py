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


class Onoma2DSP:
    """オノマトペから音声処理までのエンドツーエンドクラス"""

    def __init__(self, model_path: str, scaler_path: str = None,
                 device: str = 'cpu', sample_rate: int = 44100,
                 amplification_factor: float = 5.0):
        """
        Args:
            model_path: 学習済みモデルのパス
            scaler_path: スケーラーのパス
            device: デバイス ('cpu' or 'cuda')
            sample_rate: サンプリングレート
            amplification_factor: モデル出力の増幅率（デフォルト5.0）
        """
        self.device = device
        self.sample_rate = sample_rate
        self.amplification_factor = amplification_factor

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

    def process(self, onomatopoeia: str, input_audio_path: str,
               output_audio_path: str, verbose: bool = True) -> dict:
        """
        オノマトペから音声処理まで実行

        Args:
            onomatopoeia: オノマトペ（カタカナ）
            input_audio_path: 入力音声ファイルパス
            output_audio_path: 出力音声ファイルパス
            verbose: 詳細表示するか

        Returns:
            処理結果の辞書
        """
        if verbose:
            print("\n" + "="*60)
            print(f"ONOMATOPOEIA TO DSP PROCESSING")
            print("="*60)
            print(f"\nOnomatopoeia: {onomatopoeia}")
            print(f"Input audio: {input_audio_path}")
            print(f"Output audio: {output_audio_path}")
            print()

        # ステップ1: 特徴量抽出
        if verbose:
            print("[1/3] Extracting features from onomatopoeia...")

        phonemes = self.katakana_converter.convert(onomatopoeia)
        moras = self.mora_converter.convert(phonemes)
        features = self.feature_extractor.extract_features(phonemes, moras)

        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))[0]

        if verbose:
            print(f"  Phonemes: {phonemes}")
            print(f"  Moras: {[''.join(m) for m in moras]}")
            print(f"  Feature vector: 38 dimensions")

        # ステップ2: DSPパラメータ推論
        if verbose:
            print("\n[2/3] Predicting DSP parameters...")

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            normalized_params = self.model(features_tensor).cpu().numpy()[0]

        # モデル出力を増幅（より大きな変化を生み出すため）
        normalized_params = np.clip(normalized_params * self.amplification_factor, -1.0, 1.0)

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

        # ステップ3: 音声処理
        if verbose:
            print(f"\n[3/3] Applying DSP effects to audio...")

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
            'onomatopoeia': onomatopoeia,
            'phonemes': phonemes,
            'moras': [''.join(m) for m in moras],
            'normalized_params': normalized_params.tolist(),
            'mapped_params': mapped_params,
            'input_audio': input_audio_path,
            'output_audio': output_audio_path
        }

        return result


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='オノマトペから音声にDSPエフェクトを適用'
    )
    parser.add_argument(
        '--onomatopoeia', '-o',
        type=str,
        required=True,
        help='オノマトペ（カタカナ）'
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
        default='../models/saved_model.pth',
        help='学習済みモデルのパス'
    )
    parser.add_argument(
        '--scaler', '-s',
        type=str,
        default='../models/scaler.pkl',
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
            args.onomatopoeia,
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
