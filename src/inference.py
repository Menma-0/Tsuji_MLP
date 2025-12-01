"""
オノマトペからDSPパラメータを推論するCLIツール
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


class OnomatopoeiaInference:
    """オノマトペ推論クラス"""

    def __init__(self, model_path: str, scaler_path: str = None,
                 device: str = 'cpu'):
        """
        Args:
            model_path: 学習済みモデルのパス
            scaler_path: スケーラーのパス
            device: デバイス ('cpu' or 'cuda')
        """
        self.device = device
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

        self.mapper = DSPParameterMapping()

    def extract_features(self, onomatopoeia: str) -> np.ndarray:
        """
        オノマトペから特徴量を抽出

        Args:
            onomatopoeia: オノマトペ（カタカナ）

        Returns:
            特徴量ベクトル (38,)
        """
        # カタカナ → 音素列 → モーラ列
        phonemes = self.katakana_converter.convert(onomatopoeia)
        moras = self.mora_converter.convert(phonemes)

        # 特徴量抽出
        features = self.feature_extractor.extract_features(phonemes, moras)

        # スケーリング
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))[0]

        return features

    def predict(self, onomatopoeia: str) -> dict:
        """
        オノマトペからDSPパラメータを推論

        Args:
            onomatopoeia: オノマトペ（カタカナ）

        Returns:
            推論結果の辞書
        """
        # 特徴量抽出
        features = self.extract_features(onomatopoeia)

        # 推論
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            normalized_params = self.model(features_tensor).cpu().numpy()[0]

        # パラメータマッピング
        mapped_params = self.mapper.map_parameters(normalized_params)

        # 結果を整形
        result = {
            'onomatopoeia': onomatopoeia,
            'normalized_params': normalized_params.tolist(),
            'mapped_params': mapped_params
        }

        return result

    def print_result(self, result: dict, verbose: bool = False):
        """
        結果を表示

        Args:
            result: 推論結果
            verbose: 詳細表示するか
        """
        print("\n" + "="*60)
        print(f"ONOMATOPOEIA: {result['onomatopoeia']}")
        print("="*60)

        if verbose:
            print("\nNormalized Parameters (-1 to +1):")
            param_names = [
                'gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                'eq_high', 'eq_presence', 'transient_attack',
                'transient_sustain', 'time_stretch'
            ]
            for i, (name, value) in enumerate(zip(param_names, result['normalized_params'])):
                print(f"  {name:<20}: {value:>7.3f}")

        print("\nMapped DSP Parameters:")
        for key, value in result['mapped_params'].items():
            if 'db' in key or 'ratio' in key:
                unit = ''
                if 'db' in key:
                    unit = ' dB'
                elif 'ratio' in key:
                    unit = 'x'
                print(f"  {key:<25}: {value:>7.3f}{unit}")
            else:
                print(f"  {key:<25}: {value:>7.3f}")

        print("="*60 + "\n")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='オノマトペからDSPパラメータを推論'
    )
    parser.add_argument(
        '--onomatopoeia', '-o',
        type=str,
        required=True,
        help='オノマトペ（カタカナ）'
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
        '--verbose', '-v',
        action='store_true',
        help='詳細表示'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='デバイス'
    )

    args = parser.parse_args()

    # モデルとスケーラーのパスを確認
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first using train.py")
        return

    # 推論実行
    try:
        inference = OnomatopoeiaInference(
            model_path=args.model,
            scaler_path=args.scaler,
            device=args.device
        )

        result = inference.predict(args.onomatopoeia)
        inference.print_result(result, verbose=args.verbose)

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
