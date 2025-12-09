"""モデルの全精度評価"""
import torch
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')

from src.data.pair_data_loader import PairDataset
from src.models.mlp_model import Onoma2DSPMLP


def evaluate(model, X, y, device, set_name):
    """評価を実行"""
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        y_pred = model(X_tensor).cpu().numpy()

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    sign_acc = np.mean(np.sign(y) == np.sign(y_pred))

    print(f'\n{set_name} ({len(X)} samples)')
    print('=' * 80)
    print(f'Overall - MSE: {mse:.6f}, R2: {r2:.4f}, Sign Acc: {sign_acc:.4f} ({sign_acc*100:.1f}%)')

    param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                   'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']

    print(f"\n{'Parameter':<15} {'MSE':>10} {'R2':>10} {'Sign Acc':>10} {'Corr':>10}")
    print('-' * 55)

    for i, name in enumerate(param_names):
        mse_i = mean_squared_error(y[:, i], y_pred[:, i])
        r2_i = r2_score(y[:, i], y_pred[:, i])
        sign_acc_i = np.mean(np.sign(y[:, i]) == np.sign(y_pred[:, i]))
        corr = np.corrcoef(y[:, i], y_pred[:, i])[0, 1]
        print(f'{name:<15} {mse_i:>10.4f} {r2_i:>10.4f} {sign_acc_i:>10.4f} {corr:>10.4f}')

    return y_pred


def main():
    print('=' * 80)
    print('MODEL EVALUATION - ALL METRICS')
    print('=' * 80)

    # データロード
    print('\nLoading data...')
    dataset = PairDataset()
    dataset.load_data('data/training_pairs_balanced.csv')
    dataset.prepare_features(verbose=False)
    dataset.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    X_train, y_train = dataset.get_train_data()
    X_val, y_val = dataset.get_val_data()
    X_test, y_test = dataset.get_test_data()

    # モデルロード
    print('Loading model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=64, use_tanh=False)
    model.load_state_dict(torch.load('models/pair_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # 各セットで評価
    evaluate(model, X_train, y_train, device, 'TRAIN SET')
    evaluate(model, X_val, y_val, device, 'VALIDATION SET')
    evaluate(model, X_test, y_test, device, 'TEST SET')

    print('\n' + '=' * 80)


if __name__ == '__main__':
    main()
