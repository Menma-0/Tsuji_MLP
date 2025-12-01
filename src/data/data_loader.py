"""
オノマトペデータセットのローダーと前処理パイプライン
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import sys
import os

# 親ディレクトリのモジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.katakana_to_phoneme import KatakanaToPhoneme
from preprocessing.phoneme_to_mora import PhonemeToMora
from preprocessing.feature_extractor import OnomatopoeiaFeatureExtractor


class OnomatopoeiaDataset:
    """
    オノマトペデータセットを管理するクラス
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Args:
            data_path: データセットのパス（CSVファイル）
        """
        self.data_path = data_path
        self.katakana_converter = KatakanaToPhoneme()
        self.mora_converter = PhonemeToMora()
        self.feature_extractor = OnomatopoeiaFeatureExtractor()
        self.scaler = StandardScaler()

        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_data(self, data: pd.DataFrame):
        """
        データをロードする

        Args:
            data: データフレーム
                必須カラム:
                - sound_id: 音源ID
                - onomatopoeia_katakana: オノマトペ（カタカナ）
                - dsp_target_0 〜 dsp_target_9: DSPパラメータ（10次元）
                オプション:
                - confidence_score: 自己信頼スコア
                - acceptance_score: 他者受容スコア
        """
        self.data = data
        print(f"Loaded {len(data)} samples")

    def filter_by_score(self, confidence_threshold: float = 4.0,
                       acceptance_threshold: float = 4.0):
        """
        スコアでデータをフィルタリングする

        Args:
            confidence_threshold: 自己信頼スコアのしきい値
            acceptance_threshold: 他者受容スコアのしきい値
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        original_size = len(self.data)

        if 'confidence_score' in self.data.columns:
            self.data = self.data[self.data['confidence_score'] >= confidence_threshold]

        if 'acceptance_score' in self.data.columns:
            self.data = self.data[self.data['acceptance_score'] >= acceptance_threshold]

        filtered_size = len(self.data)
        print(f"Filtered: {original_size} -> {filtered_size} samples")

    def extract_features(self) -> np.ndarray:
        """
        オノマトペから特徴量を抽出する

        Returns:
            特徴量行列 (N, 38)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        features = []

        for idx, row in self.data.iterrows():
            onomatopoeia = row['onomatopoeia_katakana']

            # カタカナ → 音素列 → モーラ列
            phonemes = self.katakana_converter.convert(onomatopoeia)
            moras = self.mora_converter.convert(phonemes)

            # 特徴量抽出
            feature_vec = self.feature_extractor.extract_features(phonemes, moras)
            features.append(feature_vec)

        return np.array(features)

    def extract_targets(self) -> np.ndarray:
        """
        DSPパラメータ（教師ラベル）を抽出する

        Returns:
            DSPパラメータ行列 (N, 10)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # dsp_target_0 〜 dsp_target_9 のカラムを抽出
        target_cols = [f'dsp_target_{i}' for i in range(10)]

        return self.data[target_cols].values

    def split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                  test_ratio: float = 0.15, random_state: int = 42):
        """
        データをtrain/val/testに分割する
        sound_id単位で分割することで、同じ音源が複数のセットに入らないようにする

        Args:
            train_ratio: トレーニングデータの比率
            val_ratio: バリデーションデータの比率
            test_ratio: テストデータの比率
            random_state: 乱数シード
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # 特徴量とターゲットを抽出
        print("Extracting features...")
        X = self.extract_features()
        y = self.extract_targets()

        # sound_idごとに分割
        unique_sound_ids = self.data['sound_id'].unique()
        n_sounds = len(unique_sound_ids)

        print(f"Total unique sound IDs: {n_sounds}")

        # sound_idをシャッフルして分割
        train_ids, temp_ids = train_test_split(
            unique_sound_ids,
            test_size=(val_ratio + test_ratio),
            random_state=random_state
        )

        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_state
        )

        # インデックスを取得
        train_mask = self.data['sound_id'].isin(train_ids)
        val_mask = self.data['sound_id'].isin(val_ids)
        test_mask = self.data['sound_id'].isin(test_ids)

        # データを分割
        X_train_raw = X[train_mask]
        X_val_raw = X[val_mask]
        X_test_raw = X[test_mask]

        self.y_train = y[train_mask]
        self.y_val = y[val_mask]
        self.y_test = y[test_mask]

        # 特徴量をスケーリング（trainでfitし、val/testに適用）
        self.scaler.fit(X_train_raw)
        self.X_train = self.scaler.transform(X_train_raw)
        self.X_val = self.scaler.transform(X_val_raw)
        self.X_test = self.scaler.transform(X_test_raw)

        print(f"Train: {len(self.X_train)} samples")
        print(f"Val: {len(self.X_val)} samples")
        print(f"Test: {len(self.X_test)} samples")

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """トレーニングデータを取得"""
        return self.X_train, self.y_train

    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """バリデーションデータを取得"""
        return self.X_val, self.y_val

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """テストデータを取得"""
        return self.X_test, self.y_test

    def process_single_onomatopoeia(self, onomatopoeia: str) -> np.ndarray:
        """
        単一のオノマトペを処理して特徴量ベクトルを返す

        Args:
            onomatopoeia: オノマトペ（カタカナ）

        Returns:
            正規化された特徴量ベクトル (38,)
        """
        # カタカナ → 音素列 → モーラ列
        phonemes = self.katakana_converter.convert(onomatopoeia)
        moras = self.mora_converter.convert(phonemes)

        # 特徴量抽出
        feature_vec = self.feature_extractor.extract_features(phonemes, moras)

        # スケーリング
        feature_vec_scaled = self.scaler.transform(feature_vec.reshape(1, -1))

        return feature_vec_scaled[0]


def create_dummy_dataset(n_samples: int = 1000, n_sounds: int = 100) -> pd.DataFrame:
    """
    ダミーデータセットを作成する（テスト用）

    Args:
        n_samples: サンプル数
        n_sounds: 音源ID数

    Returns:
        ダミーデータフレーム
    """
    np.random.seed(42)

    onomatopoeia_list = [
        'ガンガン', 'ゴロゴロ', 'ズシャーーッ', 'サラサラ',
        'パタパタ', 'ドンドン', 'キラキラ', 'フワフワ',
        'バシャバシャ', 'ピカピカ', 'ザーザー', 'ビュービュー'
    ]

    data = []

    for i in range(n_samples):
        sound_id = np.random.randint(0, n_sounds)
        onomatopoeia = np.random.choice(onomatopoeia_list)

        # ダミーのDSPパラメータ（-1〜+1）
        dsp_params = np.random.randn(10) * 0.5
        dsp_params = np.clip(dsp_params, -1, 1)

        # スコア
        confidence_score = np.random.randint(1, 6)
        acceptance_score = np.random.randint(1, 6)

        row = {
            'sound_id': sound_id,
            'onomatopoeia_katakana': onomatopoeia,
            'confidence_score': confidence_score,
            'acceptance_score': acceptance_score
        }

        for j in range(10):
            row[f'dsp_target_{j}'] = dsp_params[j]

        data.append(row)

    return pd.DataFrame(data)


def test_data_loader():
    """データローダーのテスト"""
    print("=== Data Loader Test ===\n")

    # ダミーデータセットを作成
    dummy_data = create_dummy_dataset(n_samples=500, n_sounds=50)
    print(f"Created dummy dataset: {dummy_data.shape}")
    print(f"Columns: {dummy_data.columns.tolist()}")
    print()

    # データセットをロード
    dataset = OnomatopoeiaDataset()
    dataset.load_data(dummy_data)

    # スコアでフィルタリング
    dataset.filter_by_score(confidence_threshold=4.0, acceptance_threshold=4.0)

    # データ分割
    dataset.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    # データを取得
    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_val_data()
    X_test, y_test = dataset.get_test_data()

    print(f"\nTrain X shape: {X_train.shape}, y shape: {y_train.shape}")
    print(f"Val X shape: {X_val.shape}, y shape: {y_val.shape}")
    print(f"Test X shape: {X_test.shape}, y shape: {y_test.shape}")

    # 単一オノマトペの処理テスト
    print("\n=== Single Onomatopoeia Processing ===")
    test_onoma = 'ガンガン'
    feature_vec = dataset.process_single_onomatopoeia(test_onoma)
    print(f"{test_onoma} -> Feature vector shape: {feature_vec.shape}")
    print(f"Feature vector: {feature_vec}")


if __name__ == '__main__':
    test_data_loader()
