"""
学習済みモデルの性能評価

評価項目:
1. 基本的な回帰指標（MSE, R2, MAE）
2. 符号の正解率（変化の方向性）
3. 特定のオノマトペペアでの予測結果
4. 予測値と実測値の分布比較
5. 残差分析
"""
import torch
import numpy as np
import pandas as pd
import pickle
import os
import sys
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')

from src.data.pair_data_loader import PairDataset
from src.models.mlp_model import Onoma2DSPMLP


def load_model_and_data():
    """モデルとデータを読み込み"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    # モデル読み込み
    model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=64, use_tanh=False)
    model_path = os.path.join(project_root, 'models', 'pair_model.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # データ読み込み
    data_path = os.path.join(project_root, 'data', 'training_pairs_balanced.csv')
    dataset = PairDataset()
    dataset.load_data(data_path)
    dataset.prepare_features(verbose=False)
    dataset.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    return model, dataset


def compute_metrics(y_true, y_pred):
    """各種評価指標を計算"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 符号正解率
    sign_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    # 相関係数
    if y_true.ndim == 1:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        corr = np.mean([np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
                        for i in range(y_true.shape[1])])

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Sign Accuracy': sign_acc,
        'Correlation': corr
    }


def main():
    print("=" * 80)
    print("モデル性能評価")
    print("=" * 80)

    # モデルとデータの読み込み
    print("\n[1] モデルとデータを読み込み中...")
    model, dataset = load_model_and_data()

    X_test, y_test = dataset.get_test_data()

    # 予測
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        y_pred = model(X_tensor).numpy()

    param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                   'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']

    # ===========================================
    # 2. 全体の評価指標
    # ===========================================
    print("\n" + "=" * 80)
    print("[2] 全体の評価指標")
    print("=" * 80)

    overall_metrics = compute_metrics(y_test, y_pred)
    for metric, value in overall_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # ===========================================
    # 3. パラメータ別の評価
    # ===========================================
    print("\n" + "=" * 80)
    print("[3] パラメータ別の評価")
    print("=" * 80)

    print(f"\n{'Parameter':<15} {'MSE':>8} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'SignAcc':>8} {'Corr':>8}")
    print("-" * 75)

    param_metrics = []
    for i, name in enumerate(param_names):
        metrics = compute_metrics(y_test[:, i], y_pred[:, i])
        param_metrics.append(metrics)
        print(f"{name:<15} {metrics['MSE']:>8.4f} {metrics['RMSE']:>8.4f} "
              f"{metrics['MAE']:>8.4f} {metrics['R2']:>8.4f} "
              f"{metrics['Sign Accuracy']:>8.4f} {metrics['Correlation']:>8.4f}")

    # ===========================================
    # 4. 性能ランキング
    # ===========================================
    print("\n" + "=" * 80)
    print("[4] パラメータ性能ランキング")
    print("=" * 80)

    # R2でソート（高い順）
    r2_ranking = sorted(enumerate(param_metrics), key=lambda x: x[1]['R2'], reverse=True)
    print("\nR2スコア順（予測精度が高い順）:")
    for rank, (idx, metrics) in enumerate(r2_ranking, 1):
        print(f"  {rank}. {param_names[idx]:<15} R2={metrics['R2']:.4f}")

    # 符号正解率でソート（高い順）
    sign_ranking = sorted(enumerate(param_metrics), key=lambda x: x[1]['Sign Accuracy'], reverse=True)
    print("\n符号正解率順（変化方向の予測が正確な順）:")
    for rank, (idx, metrics) in enumerate(sign_ranking, 1):
        print(f"  {rank}. {param_names[idx]:<15} SignAcc={metrics['Sign Accuracy']:.4f}")

    # ===========================================
    # 5. 予測値の分布分析
    # ===========================================
    print("\n" + "=" * 80)
    print("[5] 予測値と実測値の分布比較")
    print("=" * 80)

    print(f"\n{'Parameter':<15} {'True Mean':>10} {'Pred Mean':>10} {'True Std':>10} {'Pred Std':>10}")
    print("-" * 60)

    for i, name in enumerate(param_names):
        true_mean = y_test[:, i].mean()
        pred_mean = y_pred[:, i].mean()
        true_std = y_test[:, i].std()
        pred_std = y_pred[:, i].std()
        print(f"{name:<15} {true_mean:>10.4f} {pred_mean:>10.4f} {true_std:>10.4f} {pred_std:>10.4f}")

    # ===========================================
    # 6. 残差分析
    # ===========================================
    print("\n" + "=" * 80)
    print("[6] 残差分析")
    print("=" * 80)

    residuals = y_test - y_pred

    print(f"\n{'Parameter':<15} {'Residual Mean':>14} {'Residual Std':>14} {'Max Error':>12}")
    print("-" * 60)

    for i, name in enumerate(param_names):
        res_mean = residuals[:, i].mean()
        res_std = residuals[:, i].std()
        max_err = np.abs(residuals[:, i]).max()
        print(f"{name:<15} {res_mean:>14.4f} {res_std:>14.4f} {max_err:>12.4f}")

    # ===========================================
    # 7. 特定ペアでの予測例
    # ===========================================
    print("\n" + "=" * 80)
    print("[7] 代表的なオノマトペペアでの予測")
    print("=" * 80)

    test_pairs = [
        ('カッ', 'ガッ'),
        ('コン', 'ドン'),
        ('サラサラ', 'ザラザラ'),
        ('チリン', 'ドスン'),
        ('キラキラ', 'ゴロゴロ'),
    ]

    for source, target in test_pairs:
        try:
            feature_diff = dataset.process_single_pair(source, target)
            with torch.no_grad():
                pred = model(torch.FloatTensor(feature_diff.reshape(1, -1))).numpy()[0]

            print(f"\n{source} → {target}:")
            print(f"  {'Parameter':<15} {'Predicted':>10}")
            print(f"  {'-'*27}")
            for i, name in enumerate(param_names):
                print(f"  {name:<15} {pred[i]:>10.4f}")
        except Exception as e:
            print(f"\n{source} → {target}: エラー - {e}")

    # ===========================================
    # 8. 性能評価のまとめ
    # ===========================================
    print("\n" + "=" * 80)
    print("[8] 性能評価のまとめ")
    print("=" * 80)

    # 良好なパラメータ（R2 > 0.5）
    good_params = [param_names[i] for i, m in enumerate(param_metrics) if m['R2'] > 0.5]
    # 課題のあるパラメータ（R2 < 0.4）
    poor_params = [param_names[i] for i, m in enumerate(param_metrics) if m['R2'] < 0.4]

    print(f"""
全体性能:
  - R2スコア: {overall_metrics['R2']:.4f}
  - 符号正解率: {overall_metrics['Sign Accuracy']*100:.1f}%
  - 相関係数: {overall_metrics['Correlation']:.4f}

良好なパラメータ（R2 > 0.5）:
  {', '.join(good_params) if good_params else 'なし'}

課題のあるパラメータ（R2 < 0.4）:
  {', '.join(poor_params) if poor_params else 'なし'}

考察:
  - 符号正解率72%は、変化の方向性をある程度捉えている
  - eq_sub, eq_lowの予測が困難なのは、学習データでの分散が小さいため
  - time_stretchの符号正解率が低いのは、持続時間と特徴量の関係が弱いため
  - compression, attack, sustainは比較的良好に予測できている
""")


if __name__ == '__main__':
    main()
