"""
データセット作成支援ツール
オノマトペとDSPパラメータのペアを作成するための補助スクリプト
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import os


class DatasetCreator:
    """データセット作成クラス"""

    def __init__(self):
        """初期化"""
        self.samples = []

    def add_sample(self, sound_id: int, onomatopoeia: str,
                  dsp_params: List[float], confidence_score: int = 5,
                  acceptance_score: int = 5):
        """
        サンプルを追加

        Args:
            sound_id: 音源ID
            onomatopoeia: オノマトペ（カタカナ）
            dsp_params: DSPパラメータ（10次元リスト、-1〜+1）
            confidence_score: 自己信頼スコア（1〜5）
            acceptance_score: 他者受容スコア（1〜5）
        """
        if len(dsp_params) != 10:
            raise ValueError(f"DSP parameters must be 10-dimensional, got {len(dsp_params)}")

        sample = {
            'sound_id': sound_id,
            'onomatopoeia_katakana': onomatopoeia,
            'confidence_score': confidence_score,
            'acceptance_score': acceptance_score
        }

        for i, param in enumerate(dsp_params):
            sample[f'dsp_target_{i}'] = param

        self.samples.append(sample)

    def add_batch_from_templates(self, sound_ids: List[int],
                                 onomatopoeia_templates: Dict[str, List[float]]):
        """
        テンプレートから一括でサンプルを追加

        Args:
            sound_ids: 音源IDのリスト
            onomatopoeia_templates: オノマトペとDSPパラメータのテンプレート
                例: {
                    'ガンガン': [0.5, 0.3, 0.2, 0.5, 0.3, 0.1, -0.2, 0.8, 0.2, 0.0],
                    'サラサラ': [-0.3, -0.2, 0.0, -0.2, 0.3, 0.5, 0.6, -0.4, 0.4, 0.0]
                }
        """
        for sound_id in sound_ids:
            for onomatopoeia, dsp_params in onomatopoeia_templates.items():
                # ランダムに少しバリエーションを加える
                varied_params = [p + np.random.randn() * 0.1 for p in dsp_params]
                varied_params = [np.clip(p, -1, 1) for p in varied_params]

                # ランダムなスコア
                confidence = np.random.randint(3, 6)
                acceptance = np.random.randint(3, 6)

                self.add_sample(sound_id, onomatopoeia, varied_params,
                              confidence, acceptance)

    def save_to_csv(self, filepath: str):
        """
        データセットをCSVに保存

        Args:
            filepath: 保存先ファイルパス
        """
        df = pd.DataFrame(self.samples)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Dataset saved to: {filepath}")
        print(f"Total samples: {len(self.samples)}")

    def load_from_csv(self, filepath: str):
        """
        CSVからデータセットを読み込み

        Args:
            filepath: CSVファイルパス
        """
        df = pd.read_csv(filepath)
        self.samples = df.to_dict('records')
        print(f"Dataset loaded from: {filepath}")
        print(f"Total samples: {len(self.samples)}")


def create_example_templates() -> Dict[str, List[float]]:
    """
    例示用のオノマトペテンプレートを作成

    Returns:
        オノマトペとDSPパラメータのテンプレート辞書
    """
    # パラメータ順:
    # 0: gain, 1: compression, 2: eq_sub, 3: eq_low, 4: eq_mid,
    # 5: eq_high, 6: eq_presence, 7: transient_attack,
    # 8: transient_sustain, 9: time_stretch

    templates = {
        # 強い音（ガンガン、ドンドン）
        'ガンガン': [0.6, 0.5, 0.3, 0.5, 0.4, 0.2, -0.1, 0.8, 0.3, 0.0],
        'ドンドン': [0.7, 0.6, 0.8, 0.6, 0.2, -0.2, -0.3, 0.9, 0.2, -0.1],
        'バンバン': [0.6, 0.5, 0.5, 0.6, 0.4, 0.1, -0.2, 0.7, 0.3, 0.0],

        # 軽い音（サラサラ、パタパタ）
        'サラサラ': [-0.2, -0.3, -0.2, -0.3, 0.2, 0.5, 0.7, -0.5, 0.5, 0.1],
        'パタパタ': [0.0, -0.2, -0.4, -0.2, 0.3, 0.4, 0.5, -0.3, 0.4, 0.05],
        'カサカサ': [-0.1, -0.2, -0.3, -0.2, 0.3, 0.6, 0.6, -0.4, 0.6, 0.0],

        # 重い音（ズシャー、ゴロゴロ）
        'ズシャー': [0.5, 0.7, 0.6, 0.7, 0.3, -0.2, -0.4, 0.6, 0.7, -0.2],
        'ゴロゴロ': [0.4, 0.5, 0.7, 0.6, 0.2, -0.3, -0.3, 0.4, 0.5, -0.1],

        # きらびやかな音（キラキラ、ピカピカ）
        'キラキラ': [-0.3, -0.4, -0.5, -0.4, 0.1, 0.7, 0.9, -0.2, 0.6, 0.15],
        'ピカピカ': [-0.2, -0.3, -0.4, -0.3, 0.2, 0.8, 0.9, -0.1, 0.5, 0.1],

        # ふわっとした音（フワフワ）
        'フワフワ': [-0.4, -0.5, -0.2, -0.1, 0.3, 0.4, 0.3, -0.7, 0.7, 0.2],

        # 激しい音（バシャバシャ、ザーザー）
        'バシャバシャ': [0.3, 0.4, 0.2, 0.3, 0.5, 0.6, 0.4, 0.5, 0.4, 0.0],
        'ザーザー': [0.4, 0.5, 0.1, 0.2, 0.4, 0.5, 0.5, 0.3, 0.6, 0.0],
    }

    return templates


def create_sample_dataset(output_path: str = 'sample_dataset.csv',
                         n_sounds: int = 50):
    """
    サンプルデータセットを作成

    Args:
        output_path: 出力ファイルパス
        n_sounds: 音源数
    """
    print("Creating sample dataset...")

    creator = DatasetCreator()

    # テンプレートを取得
    templates = create_example_templates()

    # 音源IDのリスト
    sound_ids = list(range(n_sounds))

    # テンプレートから一括追加
    creator.add_batch_from_templates(sound_ids, templates)

    # 保存
    creator.save_to_csv(output_path)

    print("\nSample dataset created successfully!")
    print(f"Onomatopoeia types: {len(templates)}")
    print(f"Sound IDs: {n_sounds}")
    print(f"Total samples: {len(creator.samples)}")


def print_template_guide():
    """テンプレートガイドを表示"""
    print("\n" + "="*60)
    print("DSP PARAMETER TEMPLATE GUIDE")
    print("="*60)

    print("\nParameter Index and Meaning:")
    print("  0: gain           - Volume (-1: quiet, +1: loud)")
    print("  1: compression    - Dynamics (-1: soft, +1: compressed)")
    print("  2: eq_sub         - Sub bass 80Hz (-1: cut, +1: boost)")
    print("  3: eq_low         - Low 250Hz (-1: cut, +1: boost)")
    print("  4: eq_mid         - Mid 1kHz (-1: cut, +1: boost)")
    print("  5: eq_high        - High 4kHz (-1: cut, +1: boost)")
    print("  6: eq_presence    - Presence 10kHz (-1: cut, +1: boost)")
    print("  7: transient_attack  - Attack (-1: soft, +1: sharp)")
    print("  8: transient_sustain - Sustain (-1: short, +1: long)")
    print("  9: time_stretch   - Duration (-1: faster, +1: slower)")

    print("\nExample Templates:")
    templates = create_example_templates()

    for onoma, params in list(templates.items())[:3]:
        print(f"\n  '{onoma}': {params}")

    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    # ガイドを表示
    print_template_guide()

    # サンプルデータセットを作成
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'sample_dataset.csv')
    create_sample_dataset(output_path, n_sounds=100)
