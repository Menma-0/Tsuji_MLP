"""
オノマトペペアデータでモデルを学習

入力: オノマトペ特徴量の差分 (target - source) (38次元)
出力: DSPパラメータの差分 (10次元)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from src.data.pair_data_loader import PairDataset
from src.models.mlp_model import Onoma2DSPMLP


def compute_sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """符号の正解率を計算"""
    correct = np.sum(np.sign(y_true) == np.sign(y_pred))
    total = y_true.size
    return correct / total


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device
) -> tuple:
    """モデルを評価"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        y_pred = model(X_tensor).cpu().numpy()

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    sign_acc = compute_sign_accuracy(y, y_pred)

    return mse, r2, sign_acc, y_pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 200,
    patience: int = 20
) -> tuple:
    """
    モデルを学習（Early Stopping付き）
    """
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # ログ出力
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ベストモデルをロード
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


def main():
    print("=" * 80)
    print("TRAINING WITH ONOMATOPOEIA PAIR DATA")
    print("=" * 80)

    # パスの設定
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', 'training_pairs_balanced.csv')

    # 1. データセットのロード
    print("\n[1] Loading pair dataset...")
    dataset = PairDataset()
    dataset.load_data(data_path)

    # 2. 特徴量の準備
    print("\n[2] Preparing features...")
    dataset.prepare_features(verbose=True)

    # 3. データの分割
    print("\n[3] Splitting data...")
    dataset.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_val_data()
    X_test, y_test = dataset.get_test_data()

    # 4. PyTorchデータローダーの作成
    print("\n[4] Creating data loaders...")
    batch_size = 64

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

    # 5. モデルの準備
    print("\n[5] Preparing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 入力: 38次元（特徴量差分）、出力: 10次元（DSP差分）
    # Tanhは使用しない（差分なので-1〜+1に収まらない可能性がある）
    model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=64, use_tanh=False)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 6. 学習
    print("\n[6] Training model...")
    print("=" * 80)
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, device,
        epochs=300, patience=30
    )
    print("=" * 80)

    # 7. テストセットで評価
    print("\n[7] Evaluating on test set...")
    mse, r2, sign_acc, y_pred = evaluate_model(model, X_test, y_test, device)

    print(f"\nTest Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  Sign Accuracy: {sign_acc:.4f} ({sign_acc*100:.1f}%)")

    # 8. 各パラメータごとの評価
    print("\n[8] Per-parameter evaluation:")
    print("=" * 80)
    param_names = [
        'gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
        'eq_high', 'eq_presence', 'transient_attack', 'transient_sustain', 'time_stretch'
    ]

    print(f"{'Parameter':<20} {'MSE':<12} {'R2':<12} {'Sign Acc':<12}")
    print("-" * 56)

    for i, param_name in enumerate(param_names):
        mse_i = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2_i = r2_score(y_test[:, i], y_pred[:, i])
        sign_acc_i = compute_sign_accuracy(y_test[:, i], y_pred[:, i])
        print(f"{param_name:<20} {mse_i:<12.6f} {r2_i:<12.4f} {sign_acc_i:<12.4f}")

    # 9. モデルとスケーラーの保存
    print("\n[9] Saving model and scaler...")
    os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)

    model_path = os.path.join(project_root, 'models', 'pair_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    scaler_path = os.path.join(project_root, 'models', 'pair_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(dataset.scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # 10. サマリー
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\nFinal Summary:")
    print(f"  Dataset: {len(dataset.df)} pairs")
    print(f"  Model: MLP (38 -> 64 -> 10)")
    print(f"  Test MSE: {mse:.6f}")
    print(f"  Test R2: {r2:.4f}")
    print(f"  Sign Accuracy: {sign_acc:.4f}")
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


if __name__ == '__main__':
    main()
