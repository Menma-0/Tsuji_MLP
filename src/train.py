"""
オノマトペ→DSPパラメータ推論モデルの学習・評価スクリプト
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import os
import sys
import pickle

# 親ディレクトリのモジュールをインポート
sys.path.append(os.path.dirname(__file__))

from data.data_loader import OnomatopoeiaDataset, create_dummy_dataset
from models.mlp_model import Onoma2DSPMLP, SklearnMLPWrapper


class Trainer:
    """PyTorchモデルの学習クラス"""

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Args:
            model: PyTorchモデル
            device: デバイス ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module) -> float:
        """
        1エポックの学習

        Args:
            train_loader: トレーニングデータローダー
            optimizer: オプティマイザ
            criterion: 損失関数

        Returns:
            平均損失
        """
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # 順伝播
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)

            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, data_loader: DataLoader, criterion: nn.Module) -> float:
        """
        評価

        Args:
            data_loader: データローダー
            criterion: 損失関数

        Returns:
            平均損失
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item()

        return total_loss / len(data_loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             n_epochs: int, learning_rate: float = 1e-3):
        """
        学習ループ

        Args:
            train_loader: トレーニングデータローダー
            val_loader: バリデーションデータローダー
            n_epochs: エポック数
            learning_rate: 学習率
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.evaluate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{n_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # ベストモデルを保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()

        # ベストモデルをロード
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nBest validation loss: {best_val_loss:.4f}")

    def plot_losses(self, save_path: str = None):
        """
        損失の推移をプロット

        Args:
            save_path: 保存パス（Noneなら表示のみ）
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            print(f"Loss plot saved to {save_path}")
        else:
            plt.show()


class Evaluator:
    """モデル評価クラス"""

    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                      param_names: list = None) -> dict:
        """
        モデルを評価する

        Args:
            y_true: 正解データ (N, 10)
            y_pred: 予測データ (N, 10)
            param_names: パラメータ名のリスト

        Returns:
            評価指標の辞書
        """
        if param_names is None:
            param_names = [
                'gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                'eq_high', 'eq_presence', 'transient_attack',
                'transient_sustain', 'time_stretch'
            ]

        results = {}

        # 全体のMSEとR2
        overall_mse = mean_squared_error(y_true, y_pred)
        overall_r2 = r2_score(y_true, y_pred)

        results['overall_mse'] = overall_mse
        results['overall_r2'] = overall_r2

        # 各次元ごとのMSEとR2
        results['per_dimension'] = {}

        for i, name in enumerate(param_names):
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            r2 = r2_score(y_true[:, i], y_pred[:, i])

            results['per_dimension'][name] = {
                'mse': mse,
                'r2': r2
            }

        # 符号の正解率（方向の正解率）
        sign_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
        results['sign_accuracy'] = sign_accuracy

        return results

    @staticmethod
    def print_results(results: dict):
        """
        評価結果を表示

        Args:
            results: 評価指標の辞書
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        print(f"\nOverall Metrics:")
        print(f"  MSE: {results['overall_mse']:.4f}")
        print(f"  R2:  {results['overall_r2']:.4f}")
        print(f"  Sign Accuracy: {results['sign_accuracy']:.4f}")

        print(f"\nPer-Dimension Metrics:")
        print(f"{'Parameter':<20} {'MSE':<10} {'R2':<10}")
        print("-" * 40)

        for name, metrics in results['per_dimension'].items():
            print(f"{name:<20} {metrics['mse']:<10.4f} {metrics['r2']:<10.4f}")

        print("="*60 + "\n")


def train_pytorch_model(dataset: OnomatopoeiaDataset, n_epochs: int = 100,
                       batch_size: int = 16, learning_rate: float = 1e-3):
    """
    PyTorchモデルを学習する

    Args:
        dataset: データセット
        n_epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率

    Returns:
        学習済みモデル、トレーナー、評価結果
    """
    print("\n" + "="*60)
    print("TRAINING PYTORCH MODEL")
    print("="*60 + "\n")

    # データを取得
    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_val_data()
    X_test, y_test = dataset.get_test_data()

    # PyTorchテンソルに変換
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # モデルを作成
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=32, use_tanh=True)

    # 学習
    trainer = Trainer(model, device=device)
    trainer.train(train_loader, val_loader, n_epochs=n_epochs,
                 learning_rate=learning_rate)

    # テストデータで評価
    print("\n" + "="*60)
    print("EVALUATING ON TEST DATA")
    print("="*60)

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_test_tensor).cpu().numpy()

    evaluator = Evaluator()
    results = evaluator.evaluate_model(y_test, y_pred)
    evaluator.print_results(results)

    return model, trainer, results, dataset


def train_sklearn_model(dataset: OnomatopoeiaDataset, max_iter: int = 500):
    """
    scikit-learnモデルを学習する

    Args:
        dataset: データセット
        max_iter: 最大イテレーション数

    Returns:
        学習済みモデル、評価結果
    """
    print("\n" + "="*60)
    print("TRAINING SCIKIT-LEARN MODEL")
    print("="*60 + "\n")

    # データを取得
    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()

    # モデルを作成して学習
    model = SklearnMLPWrapper(hidden_layer_sizes=(32,), max_iter=max_iter)
    model.fit(X_train, y_train)

    # 評価
    print("\n" + "="*60)
    print("EVALUATING ON TEST DATA")
    print("="*60)

    y_pred = model.predict(X_test)

    evaluator = Evaluator()
    results = evaluator.evaluate_model(y_test, y_pred)
    evaluator.print_results(results)

    return model, results


def main():
    """メイン関数"""
    print("\n" + "="*60)
    print("ONOMATOPOEIA TO DSP PARAMETER PREDICTION")
    print("="*60 + "\n")

    # ダミーデータセットを作成
    print("Creating dummy dataset...")
    dummy_data = create_dummy_dataset(n_samples=1000, n_sounds=100)

    # データセットをロード
    dataset = OnomatopoeiaDataset()
    dataset.load_data(dummy_data)

    # スコアでフィルタリング
    dataset.filter_by_score(confidence_threshold=4.0, acceptance_threshold=4.0)

    # データ分割
    dataset.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    # PyTorchモデルを学習
    pytorch_model, trainer, pytorch_results, dataset = train_pytorch_model(
        dataset, n_epochs=100, batch_size=16, learning_rate=1e-3
    )

    # 損失の推移をプロット
    # trainer.plot_losses(save_path='loss_plot.png')

    # scikit-learnモデルを学習（比較用）
    sklearn_model, sklearn_results = train_sklearn_model(dataset, max_iter=500)

    # 結果の比較
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"\nPyTorch Model:")
    print(f"  Overall R2: {pytorch_results['overall_r2']:.4f}")
    print(f"  Overall MSE: {pytorch_results['overall_mse']:.4f}")

    print(f"\nscikit-learn Model:")
    print(f"  Overall R2: {sklearn_results['overall_r2']:.4f}")
    print(f"  Overall MSE: {sklearn_results['overall_mse']:.4f}")

    print("\n" + "="*60 + "\n")

    # モデルとスケーラーを保存
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, 'saved_model.pth')
    scaler_path = os.path.join(save_dir, 'scaler.pkl')

    # PyTorchモデルを保存
    torch.save(pytorch_model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # スケーラーを保存
    with open(scaler_path, 'wb') as f:
        pickle.dump(dataset.scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    print("\nYou can now use inference.py to make predictions!")


if __name__ == '__main__':
    main()
