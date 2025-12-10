"""モデル性能の簡易評価"""
import torch
import numpy as np
import sys
import os
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')

from src.data.pair_data_loader import PairDataset
from src.models.mlp_model import Onoma2DSPMLP

# モデルとデータ読み込み
model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=64, use_tanh=False)
model.load_state_dict(torch.load('models/pair_model.pth', map_location='cpu', weights_only=True))
model.eval()

dataset = PairDataset()
dataset.load_data('data/training_pairs_balanced.csv')
dataset.prepare_features(verbose=False)
dataset.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

X_test, y_test = dataset.get_test_data()

with torch.no_grad():
    y_pred = model(torch.FloatTensor(X_test)).numpy()

param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
               'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']

results = []
for i, name in enumerate(param_names):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    sign_acc = np.mean(np.sign(y_test[:, i]) == np.sign(y_pred[:, i]))
    results.append((name, r2, sign_acc))

# R2でソート
results.sort(key=lambda x: x[1], reverse=True)

print('| 順位 | パラメータ | R2 | 符号正解率 |')
print('|-----|------------|-------|-------|')
for rank, (name, r2, sign_acc) in enumerate(results, 1):
    print(f'| {rank} | {name} | {r2:.3f} | {sign_acc*100:.1f}% |')
