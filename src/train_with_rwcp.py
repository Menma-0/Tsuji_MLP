"""
RWCPデータセットでモデルを学習
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import sys

# プロジェクトルートを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import OnomatopoeiaDataset
from src.models.mlp_model import Onoma2DSPMLP


def compute_sign_accuracy(y_true, y_pred):
    """符号の正解率を計算"""
    correct = np.sum(np.sign(y_true) == np.sign(y_pred))
    total = y_true.size
    return correct / total


def evaluate_model(model, X, y, device):
    """モデルを評価"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        y_pred = model(X_tensor).cpu().numpy()

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    sign_acc = compute_sign_accuracy(y, y_pred)

    return mse, r2, sign_acc, y_pred


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=200):
    """モデルを学習"""
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward
            output = model(batch_x)
            loss = criterion(output, batch_y)

            # Backward
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

        # ログ出力
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses


def main():
    print("=" * 80)
    print("TRAINING WITH RWCP-SSD-ONOMATOPOEIA DATASET")
    print("=" * 80)

    # 1. データセットのロード
    print("\n[1] Loading RWCP dataset...")
    import pandas as pd
    df = pd.read_csv('../data/rwcp_dataset.csv')

    dataset = OnomatopoeiaDataset()
    dataset.load_data(df)
    print(f"Total samples: {len(dataset.data)}")
    print(f"Unique onomatopoeia: {dataset.data['onomatopoeia_katakana'].nunique()}")

    # 2. データの分割とスケーリング
    print("\n[2] Splitting and scaling data...")
    dataset.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    X_train_scaled = dataset.X_train
    y_train = dataset.y_train
    X_val_scaled = dataset.X_val
    y_val = dataset.y_val
    X_test_scaled = dataset.X_test
    y_test = dataset.y_test

    print(f"Train samples: {len(X_train_scaled)}")
    print(f"Val samples: {len(X_val_scaled)}")
    print(f"Test samples: {len(X_test_scaled)}")

    # 3. PyTorchデータローダーの作成
    print("\n[3] Creating data loaders...")
    batch_size = 64
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. モデルの準備
    print("\n[4] Preparing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=32, use_tanh=True)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 5. 学習
    print("\n[5] Training model...")
    print("=" * 80)
    epochs = 200
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs
    )
    print("=" * 80)

    # 6. テストセットで評価
    print("\n[6] Evaluating on test set...")
    mse, r2, sign_acc, y_pred = evaluate_model(model, X_test_scaled, y_test, device)

    print(f"\nTest Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  Sign Accuracy: {sign_acc:.4f} ({sign_acc*100:.1f}%)")

    # 7. 各パラメータごとの評価
    print("\n[7] Per-parameter evaluation:")
    print("=" * 80)
    param_names = [
        'gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
        'eq_high', 'eq_presence', 'transient_attack', 'transient_sustain', 'time_stretch'
    ]

    print(f"{'Parameter':<20} {'MSE':<12} {'R2':<12}")
    print("-" * 80)

    for i, param_name in enumerate(param_names):
        mse_i = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2_i = r2_score(y_test[:, i], y_pred[:, i])
        print(f"{param_name:<20} {mse_i:<12.6f} {r2_i:<12.4f}")

    # 8. モデルとスケーラーの保存
    print("\n[8] Saving model and scaler...")
    os.makedirs('../models', exist_ok=True)

    # モデル保存
    model_path = '../models/rwcp_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # スケーラー保存
    scaler_path = '../models/rwcp_scaler.pkl'
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(dataset.scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\nFinal Summary:")
    print(f"  Dataset: RWCP-SSD-Onomatopoeia ({len(dataset.data)} samples)")
    print(f"  Model: MLP (38 -> 32 -> 10)")
    print(f"  Test MSE: {mse:.6f}")
    print(f"  Test R2: {r2:.4f}")
    print(f"  Sign Accuracy: {sign_acc:.4f}")
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


if __name__ == '__main__':
    main()
