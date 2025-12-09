"""
オノマトペペアとDSP差分の学習データ用ローダー

入力: source_onomatopoeia, target_onomatopoeia
出力: dsp_diff (10次元)

モデルへの入力は2つのオノマトペの特徴量差分 (target - source)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.katakana_to_phoneme import KatakanaToPhoneme
from preprocessing.phoneme_to_mora import PhonemeToMora
from preprocessing.feature_extractor import OnomatopoeiaFeatureExtractor


class PairDataset:
    """
    オノマトペペア学習データセット

    学習データの形式:
    - 入力: source_onomaとtarget_onomaの特徴量差分 (38次元)
    - 出力: DSPパラメータ差分 (10次元)
    """

    def __init__(self):
        self.katakana_converter = KatakanaToPhoneme()
        self.mora_converter = PhonemeToMora()
        self.feature_extractor = OnomatopoeiaFeatureExtractor()

        self.scaler = StandardScaler()

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # 特徴量キャッシュ（同じオノマトペの特徴量を再計算しないため）
        self._feature_cache = {}

    def _extract_onoma_features(self, onomatopoeia: str) -> np.ndarray:
        """
        オノマトペから38次元の特徴量を抽出（キャッシュ付き）
        """
        if onomatopoeia in self._feature_cache:
            return self._feature_cache[onomatopoeia]

        phonemes = self.katakana_converter.convert(onomatopoeia)
        moras = self.mora_converter.convert(phonemes)
        features = self.feature_extractor.extract_features(phonemes, moras)

        self._feature_cache[onomatopoeia] = features
        return features

    def load_data(self, csv_path: str):
        """
        ペアデータCSVを読み込み

        Args:
            csv_path: training_pairs_*.csv のパス
        """
        print(f"Loading pair data from: {csv_path}")
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"  Total pairs: {len(self.df)}")
        print(f"  Unique source onomatopoeia: {self.df['source_onomatopoeia'].nunique()}")
        print(f"  Unique target onomatopoeia: {self.df['target_onomatopoeia'].nunique()}")

    def prepare_features(self, verbose: bool = True):
        """
        特徴量を準備（オノマトペ→特徴量変換）
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if verbose:
            print("Extracting features from onomatopoeia pairs...")

        X_list = []
        y_list = []

        dsp_diff_cols = [f'dsp_diff_{i}' for i in range(10)]

        for idx, row in self.df.iterrows():
            source_onoma = row['source_onomatopoeia']
            target_onoma = row['target_onomatopoeia']

            # 特徴量を抽出
            source_features = self._extract_onoma_features(source_onoma)
            target_features = self._extract_onoma_features(target_onoma)

            # 特徴量差分を計算
            feature_diff = target_features - source_features

            X_list.append(feature_diff)
            y_list.append(row[dsp_diff_cols].values)

            if verbose and (idx + 1) % 10000 == 0:
                print(f"  Processed: {idx + 1} pairs")

        self.X_raw = np.array(X_list, dtype=np.float32)
        self.y_raw = np.array(y_list, dtype=np.float32)

        if verbose:
            print(f"  Feature shape: {self.X_raw.shape}")
            print(f"  Target shape: {self.y_raw.shape}")
            print(f"  Cached {len(self._feature_cache)} unique onomatopoeia features")

    def split_data(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ):
        """
        データをtrain/val/testに分割
        """
        if self.X_raw is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")

        # まずtrain+valとtestに分割
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            self.X_raw, self.y_raw,
            test_size=test_ratio,
            random_state=random_state
        )

        # trainとvalに分割
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio_adjusted,
            random_state=random_state
        )

        # スケーラーをtrainデータでfit
        self.scaler.fit(X_train)

        # スケーリング適用
        self.X_train = self.scaler.transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.X_test = self.scaler.transform(X_test)

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        print(f"Data split:")
        print(f"  Train: {len(self.X_train)} samples")
        print(f"  Val: {len(self.X_val)} samples")
        print(f"  Test: {len(self.X_test)} samples")

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """トレーニングデータを取得"""
        return self.X_train, self.y_train

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """バリデーションデータを取得"""
        return self.X_val, self.y_val

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """テストデータを取得"""
        return self.X_test, self.y_test

    def process_single_pair(
        self,
        source_onoma: str,
        target_onoma: str
    ) -> np.ndarray:
        """
        単一のオノマトペペアを処理して特徴量差分を返す

        Args:
            source_onoma: 元のオノマトペ
            target_onoma: 目標のオノマトペ

        Returns:
            スケーリング済みの特徴量差分 (38次元)
        """
        source_features = self._extract_onoma_features(source_onoma)
        target_features = self._extract_onoma_features(target_onoma)

        feature_diff = target_features - source_features
        feature_diff_scaled = self.scaler.transform(feature_diff.reshape(1, -1))

        return feature_diff_scaled[0]


def test_pair_data_loader():
    """ペアデータローダーのテスト"""
    print("=" * 60)
    print("Pair Data Loader Test")
    print("=" * 60)

    # データセットを作成
    dataset = PairDataset()

    # テスト用にサンプルデータを作成
    test_data = pd.DataFrame({
        'source_onomatopoeia': ['カッ', 'コン', 'サラサラ', 'チリン', 'ドン'],
        'target_onomatopoeia': ['ガッ', 'ドン', 'ザラザラ', 'ゴロゴロ', 'コン'],
        'dsp_diff_0': [0.1, 0.2, 0.3, 0.4, -0.2],
        'dsp_diff_1': [0.05, 0.1, 0.15, 0.2, -0.1],
        'dsp_diff_2': [-0.1, -0.2, -0.3, -0.4, 0.2],
        'dsp_diff_3': [0.0, 0.1, 0.2, 0.3, -0.1],
        'dsp_diff_4': [0.1, 0.0, -0.1, -0.2, 0.2],
        'dsp_diff_5': [-0.2, -0.1, 0.0, 0.1, 0.2],
        'dsp_diff_6': [0.15, 0.05, -0.05, -0.15, 0.1],
        'dsp_diff_7': [0.3, 0.2, 0.1, 0.0, -0.1],
        'dsp_diff_8': [-0.1, 0.0, 0.1, 0.2, -0.2],
        'dsp_diff_9': [0.0, 0.0, 0.0, 0.0, 0.0],
    })

    # 一時ファイルに保存
    temp_csv = 'temp_test_pairs.csv'
    test_data.to_csv(temp_csv, index=False, encoding='utf-8-sig')

    try:
        # データ読み込み
        dataset.load_data(temp_csv)

        # 特徴量準備
        dataset.prepare_features(verbose=True)

        # データ分割（テストなので小さいデータでも動くように調整）
        dataset.X_train = dataset.scaler.fit_transform(dataset.X_raw[:3])
        dataset.X_val = dataset.scaler.transform(dataset.X_raw[3:4])
        dataset.X_test = dataset.scaler.transform(dataset.X_raw[4:5])
        dataset.y_train = dataset.y_raw[:3]
        dataset.y_val = dataset.y_raw[3:4]
        dataset.y_test = dataset.y_raw[4:5]

        print(f"\nTrain X shape: {dataset.X_train.shape}")
        print(f"Train y shape: {dataset.y_train.shape}")

        # 単一ペア処理のテスト
        print("\nSingle pair processing test:")
        feature_diff = dataset.process_single_pair('カッ', 'ガッ')
        print(f"  Feature diff shape: {feature_diff.shape}")
        print(f"  Feature diff (first 5): {feature_diff[:5]}")

    finally:
        # 一時ファイルを削除
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

    print("\nTest completed!")


if __name__ == '__main__':
    test_pair_data_loader()
