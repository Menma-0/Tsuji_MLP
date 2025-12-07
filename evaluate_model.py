"""
差分モデルの性能評価
R^2スコア、MSE、符号正解率などで評価
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.data.data_loader import OnomatopoeiaDataset
from src.models.mlp_model import Onoma2DSPMLP


def compute_sign_accuracy(y_true, y_pred):
    """符号の正解率を計算"""
    correct = np.sum(np.sign(y_true) == np.sign(y_pred))
    total = y_true.size
    return correct / total


def evaluate_per_parameter(y_true, y_pred, param_names):
    """各パラメータごとの評価"""
    results = []

    for i, param_name in enumerate(param_names):
        y_true_param = y_true[:, i]
        y_pred_param = y_pred[:, i]

        mse = mean_squared_error(y_true_param, y_pred_param)
        mae = mean_absolute_error(y_true_param, y_pred_param)
        r2 = r2_score(y_true_param, y_pred_param)
        sign_acc = compute_sign_accuracy(y_true_param, y_pred_param)

        results.append({
            'parameter': param_name,
            'MSE': mse,
            'MAE': mae,
            'R^2': r2,
            'Sign_Accuracy': sign_acc
        })

    return pd.DataFrame(results)


def plot_predictions(y_true, y_pred, param_names, save_path='evaluation_plots.png'):
    """予測値 vs 実際の値のプロット"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=10)

        # 対角線（完全予測）
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

        # R^2スコア
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        ax.set_title(f'{param_name}\nR^2 = {r2:.3f}')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def plot_residuals(y_true, y_pred, param_names, save_path='residual_plots.png'):
    """残差プロット"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        residuals = y_pred[:, i] - y_true[:, i]

        ax.scatter(y_true[:, i], residuals, alpha=0.3, s=10)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_title(param_name)
        ax.set_xlabel('True Value')
        ax.set_ylabel('Residual (Pred - True)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Residual plot saved to: {save_path}")
    plt.close()


def main():
    print("=" * 80)
    print("MODEL PERFORMANCE EVALUATION")
    print("=" * 80)

    # 1. データセットのロード
    print("\n[1] Loading dataset...")
    df = pd.read_csv('data/rwcp_dataset.csv')

    dataset = OnomatopoeiaDataset()
    dataset.load_data(df)
    print(f"Total samples: {len(dataset.data)}")

    # 2. データ分割（学習時と同じ分割）
    print("\n[2] Splitting and extracting features...")
    X = dataset.extract_features()
    y = dataset.extract_targets()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 3. モデルのロード
    print("\n[3] Loading model...")
    device = 'cpu'
    model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=32, use_tanh=True)

    try:
        model.load_state_dict(torch.load('models/rwcp_model.pth', map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. 予測
    print("\n[4] Making predictions...")
    with torch.no_grad():
        # Training set
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_pred = model(X_train_tensor).cpu().numpy()

        # Validation set
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_pred = model(X_val_tensor).cpu().numpy()

        # Test set
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_pred = model(X_test_tensor).cpu().numpy()

    # 5. 全体的な評価
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE METRICS")
    print("=" * 80)

    # Training set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_sign_acc = compute_sign_accuracy(y_train, y_train_pred)

    print(f"\nTraining Set:")
    print(f"  MSE:            {train_mse:.6f}")
    print(f"  R^2 Score:      {train_r2:.6f}")
    print(f"  Sign Accuracy:  {train_sign_acc:.4f} ({train_sign_acc*100:.2f}%)")

    # Validation set
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_sign_acc = compute_sign_accuracy(y_val, y_val_pred)

    print(f"\nValidation Set:")
    print(f"  MSE:            {val_mse:.6f}")
    print(f"  R^2 Score:      {val_r2:.6f}")
    print(f"  Sign Accuracy:  {val_sign_acc:.4f} ({val_sign_acc*100:.2f}%)")

    # Test set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_sign_acc = compute_sign_accuracy(y_test, y_test_pred)

    print(f"\nTest Set:")
    print(f"  MSE:            {test_mse:.6f}")
    print(f"  R^2 Score:      {test_r2:.6f}")
    print(f"  Sign Accuracy:  {test_sign_acc:.4f} ({test_sign_acc*100:.2f}%)")

    # 6. パラメータごとの評価
    print("\n" + "=" * 80)
    print("PER-PARAMETER PERFORMANCE (Test Set)")
    print("=" * 80)

    param_names = [
        'gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
        'eq_high', 'eq_presence', 'transient_attack',
        'transient_sustain', 'time_stretch'
    ]

    param_results = evaluate_per_parameter(y_test, y_test_pred, param_names)
    print("\n" + param_results.to_string(index=False))

    # 7. 統計サマリ
    print("\n" + "=" * 80)
    print("PARAMETER STATISTICS")
    print("=" * 80)

    print("\nR^2 Score Statistics:")
    print(f"  Mean:    {param_results['R^2'].mean():.4f}")
    print(f"  Median:  {param_results['R^2'].median():.4f}")
    print(f"  Min:     {param_results['R^2'].min():.4f} ({param_results.loc[param_results['R^2'].idxmin(), 'parameter']})")
    print(f"  Max:     {param_results['R^2'].max():.4f} ({param_results.loc[param_results['R^2'].idxmax(), 'parameter']})")

    print("\nSign Accuracy Statistics:")
    print(f"  Mean:    {param_results['Sign_Accuracy'].mean():.4f}")
    print(f"  Median:  {param_results['Sign_Accuracy'].median():.4f}")
    print(f"  Min:     {param_results['Sign_Accuracy'].min():.4f} ({param_results.loc[param_results['Sign_Accuracy'].idxmin(), 'parameter']})")
    print(f"  Max:     {param_results['Sign_Accuracy'].max():.4f} ({param_results.loc[param_results['Sign_Accuracy'].idxmax(), 'parameter']})")

    # 8. 可視化
    print("\n[5] Creating visualizations...")

    # Create evaluation directory first
    os.makedirs('evaluation', exist_ok=True)

    # 予測 vs 実際
    plot_predictions(y_test, y_test_pred, param_names, 'evaluation/prediction_plots.png')

    # 残差プロット
    plot_residuals(y_test, y_test_pred, param_names, 'evaluation/residual_plots.png')

    # 9. 結果を保存
    print("\n[6] Saving results...")

    # CSV保存
    param_results.to_csv('evaluation/parameter_metrics.csv', index=False)
    print("Parameter metrics saved to: evaluation/parameter_metrics.csv")

    # 全体メトリクスを保存
    overall_results = pd.DataFrame({
        'Split': ['Training', 'Validation', 'Test'],
        'MSE': [train_mse, val_mse, test_mse],
        'R^2': [train_r2, val_r2, test_r2],
        'Sign_Accuracy': [train_sign_acc, val_sign_acc, test_sign_acc]
    })
    overall_results.to_csv('evaluation/overall_metrics.csv', index=False)
    print("Overall metrics saved to: evaluation/overall_metrics.csv")

    # 10. サマリー
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\n[OK] Test R^2 Score: {test_r2:.4f}")
    print(f"[OK] Test Sign Accuracy: {test_sign_acc:.4f} ({test_sign_acc*100:.2f}%)")
    print(f"[OK] Best Parameter: {param_results.loc[param_results['R^2'].idxmax(), 'parameter']} (R^2 = {param_results['R^2'].max():.4f})")
    print(f"[OK] Worst Parameter: {param_results.loc[param_results['R^2'].idxmin(), 'parameter']} (R^2 = {param_results['R^2'].min():.4f})")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - evaluation/parameter_metrics.csv")
    print("  - evaluation/overall_metrics.csv")
    print("  - evaluation/prediction_plots.png")
    print("  - evaluation/residual_plots.png")
    print("=" * 80)


if __name__ == '__main__':
    main()
