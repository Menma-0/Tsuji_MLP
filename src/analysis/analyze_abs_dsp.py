"""
DSPパラメータ差分の絶対値分析
"""
import pandas as pd
import numpy as np
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
df = pd.read_csv(os.path.join(project_root, 'data', 'training_pairs_balanced.csv'))

dsp_cols = [f'dsp_diff_{i}' for i in range(10)]
param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
               'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']

print('=' * 80)
print('DSPパラメータ差分の絶対値統計')
print('=' * 80)

print(f'\n{"Parameter":<15} {"AbsMean":>10} {"AbsMedian":>10} {"AbsStd":>10} {"Max":>10}')
print('-' * 60)

for col, name in zip(dsp_cols, param_names):
    abs_mean = df[col].abs().mean()
    abs_median = df[col].abs().median()
    abs_std = df[col].abs().std()
    max_val = df[col].abs().max()
    print(f'{name:<15} {abs_mean:>10.4f} {abs_median:>10.4f} {abs_std:>10.4f} {max_val:>10.4f}')

# 絶対値のパーセンタイル
print(f'\n\n絶対値のパーセンタイル分布:')
print(f'{"Parameter":<15} {"50%":>8} {"75%":>8} {"90%":>8} {"95%":>8} {"99%":>8}')
print('-' * 55)

for col, name in zip(dsp_cols, param_names):
    abs_col = df[col].abs()
    p50 = abs_col.quantile(0.50)
    p75 = abs_col.quantile(0.75)
    p90 = abs_col.quantile(0.90)
    p95 = abs_col.quantile(0.95)
    p99 = abs_col.quantile(0.99)
    print(f'{name:<15} {p50:>8.4f} {p75:>8.4f} {p90:>8.4f} {p95:>8.4f} {p99:>8.4f}')

# 絶対値の大きさでランキング
print(f'\n\n絶対値平均のランキング（大きい順）:')
abs_means = [(name, df[col].abs().mean()) for col, name in zip(dsp_cols, param_names)]
abs_means.sort(key=lambda x: x[1], reverse=True)
for rank, (name, val) in enumerate(abs_means, 1):
    print(f'  {rank}. {name:<15} {val:.4f}')

# 小さい値の割合
print(f'\n\n小さい差分（|diff| < 0.1）の割合:')
for col, name in zip(dsp_cols, param_names):
    small_pct = (df[col].abs() < 0.1).sum() / len(df) * 100
    print(f'  {name:<15} {small_pct:>6.1f}%')

# 大きい値の割合
print(f'\n大きい差分（|diff| > 0.5）の割合:')
for col, name in zip(dsp_cols, param_names):
    large_pct = (df[col].abs() > 0.5).sum() / len(df) * 100
    print(f'  {name:<15} {large_pct:>6.1f}%')

# 大きい値の割合
print(f'\n非常に大きい差分（|diff| > 1.0）の割合:')
for col, name in zip(dsp_cols, param_names):
    very_large_pct = (df[col].abs() > 1.0).sum() / len(df) * 100
    print(f'  {name:<15} {very_large_pct:>6.1f}%')
