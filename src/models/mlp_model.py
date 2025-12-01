"""
オノマトペからDSPパラメータを推論するMLPモデル
PyTorchとscikit-learnの両方に対応
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.neural_network import MLPRegressor
from typing import Optional


class Onoma2DSPMLP(nn.Module):
    """
    PyTorch実装のMLPモデル
    入力: 38次元のオノマトペ特徴量
    出力: 10次元のDSPパラメータ（-1〜+1に正規化）
    """

    def __init__(self, d_in: int = 38, d_out: int = 10, hidden_dim: int = 32,
                 use_tanh: bool = True):
        """
        Args:
            d_in: 入力次元数（デフォルト38）
            d_out: 出力次元数（デフォルト10）
            hidden_dim: 隠れ層のユニット数（デフォルト32）
            use_tanh: 出力層にTanhを使うかどうか（-1〜+1に制限）
        """
        super().__init__()

        layers = [
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_out)
        ]

        if use_tanh:
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル (batch_size, 38)

        Returns:
            出力テンソル (batch_size, 10)
        """
        return self.net(x)


class SklearnMLPWrapper:
    """
    scikit-learn実装のMLPモデルのラッパー
    """

    def __init__(self, hidden_layer_sizes: tuple = (32,),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 max_iter: int = 5000,
                 random_state: int = 42,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.15):
        """
        Args:
            hidden_layer_sizes: 隠れ層のサイズ（デフォルト(32,)）
            activation: 活性化関数（'relu', 'tanh', など）
            solver: 最適化アルゴリズム（'adam', 'sgd', など）
            max_iter: 最大イテレーション数
            random_state: 乱数シード
            early_stopping: early stoppingを使うかどうか
            validation_fraction: early stopping用のvalidation比率
        """
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            verbose=True
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        モデルを学習する

        Args:
            X: 入力データ (N, 38)
            y: 出力データ (N, 10)
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測を行う

        Args:
            X: 入力データ (N, 38)

        Returns:
            予測データ (N, 10)
        """
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        R^2スコアを計算する

        Args:
            X: 入力データ (N, 38)
            y: 正解データ (N, 10)

        Returns:
            R^2スコア
        """
        return self.model.score(X, y)


class DSPParameterMapping:
    """
    正規化されたDSPパラメータ(-1〜+1)を実際のdB/倍率にマッピング
    """

    @staticmethod
    def map_parameters(normalized_params: np.ndarray) -> dict:
        """
        正規化パラメータを実際の値にマッピング

        Args:
            normalized_params: 10次元の正規化パラメータ (-1〜+1)

        Returns:
            実際のパラメータ値の辞書
        """
        if len(normalized_params) != 10:
            raise ValueError(f"Expected 10 parameters, got {len(normalized_params)}")

        # パラメータ名
        param_names = [
            'gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
            'eq_high', 'eq_presence', 'transient_attack',
            'transient_sustain', 'time_stretch'
        ]

        # マッピング
        mapped = {}

        # gain: -12dB 〜 +12dB
        mapped['gain_db'] = 12.0 * normalized_params[0]

        # compression: 正規化値そのまま（DSP側で解釈）
        mapped['compression'] = normalized_params[1]

        # EQバンド: -12dB 〜 +12dB
        mapped['eq_sub_db'] = 12.0 * normalized_params[2]
        mapped['eq_low_db'] = 12.0 * normalized_params[3]
        mapped['eq_mid_db'] = 12.0 * normalized_params[4]
        mapped['eq_high_db'] = 12.0 * normalized_params[5]
        mapped['eq_presence_db'] = 12.0 * normalized_params[6]

        # トランジェント: 正規化値そのまま
        mapped['transient_attack'] = normalized_params[7]
        mapped['transient_sustain'] = normalized_params[8]

        # time_stretch: 0.5倍 〜 1.5倍
        # -1 → 0.5, 0 → 1.0, +1 → 1.5
        mapped['time_stretch_ratio'] = 1.0 + 0.5 * normalized_params[9]

        return mapped


def test_pytorch_model():
    """PyTorchモデルのテスト"""
    print("=== PyTorch Model Test ===")

    # モデルを作成
    model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=32, use_tanh=True)

    # ダミーデータ
    batch_size = 4
    dummy_input = torch.randn(batch_size, 38)

    # 推論
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Sample output: {output[0]}")

    # パラメータマッピング
    mapper = DSPParameterMapping()
    mapped = mapper.map_parameters(output[0].numpy())
    print("\nMapped parameters:")
    for key, value in mapped.items():
        print(f"  {key}: {value:.3f}")


def test_sklearn_model():
    """scikit-learnモデルのテスト"""
    print("\n=== scikit-learn Model Test ===")

    # ダミーデータ
    np.random.seed(42)
    X_train = np.random.randn(100, 38)
    y_train = np.random.randn(100, 10) * 0.5  # -1〜+1に近い範囲

    X_test = np.random.randn(20, 38)
    y_test = np.random.randn(20, 10) * 0.5

    # モデルを作成して学習
    model = SklearnMLPWrapper(hidden_layer_sizes=(32,), max_iter=100)
    print("Training...")
    model.fit(X_train, y_train)

    # 評価
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"\nTraining R^2 score: {train_score:.4f}")
    print(f"Test R^2 score: {test_score:.4f}")

    # 予測
    predictions = model.predict(X_test[:5])
    print(f"\nSample predictions shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0]}")


if __name__ == '__main__':
    test_pytorch_model()
    test_sklearn_model()
