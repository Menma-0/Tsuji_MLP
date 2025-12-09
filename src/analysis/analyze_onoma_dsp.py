"""
オノマトペごとの平均DSPパラメータの分析
（差分学習データ作成の元となったデータ）
"""
import pandas as pd
import numpy as np
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
df = pd.read_csv(os.path.join(project_root, 'data', 'rwcp_dataset_audio_based.csv'))

dsp_cols = [f'dsp_target_{i}' for i in range(10)]
param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
               'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']

print('=' * 80)
print('オノマトペごとの平均DSPパラメータ分析')
print('（差分学習データ作成の元となったデータ）')
print('=' * 80)

# ===========================================
# 1. 基本情報
# ===========================================
print('\n[1] 基本情報')
print('-' * 40)
print(f'元データのサンプル数: {len(df)}')
print(f'ユニークなオノマトペ数（全体）: {df["onomatopoeia_katakana"].nunique()}')

# サンプル数でフィルタリング
onoma_counts = df['onomatopoeia_katakana'].value_counts()
min_samples = 3
valid_onomas = onoma_counts[onoma_counts >= min_samples].index.tolist()
print(f'{min_samples}サンプル以上のオノマトペ数: {len(valid_onomas)}')

# オノマトペごとの平均DSPを計算
onoma_dsp = df.groupby('onomatopoeia_katakana')[dsp_cols].mean()
onoma_dsp = onoma_dsp.loc[valid_onomas]
onoma_dsp.columns = param_names

print(f'\n分析対象: {len(onoma_dsp)} オノマトペ')

# ===========================================
# 2. 各オノマトペのサンプル数分布
# ===========================================
print('\n[2] 各オノマトペのサンプル数分布')
print('-' * 40)
valid_counts = onoma_counts[onoma_counts >= min_samples]
print(f'サンプル数: 平均{valid_counts.mean():.1f}, 中央値{valid_counts.median():.0f}')
print(f'最小{valid_counts.min()}, 最大{valid_counts.max()}')

print(f'\nサンプル数の分布:')
bins = [3, 5, 10, 20, 50, 100, float('inf')]
labels = ['3-4', '5-9', '10-19', '20-49', '50-99', '100+']
for i in range(len(bins)-1):
    count = ((valid_counts >= bins[i]) & (valid_counts < bins[i+1])).sum()
    print(f'  {labels[i]:>8}: {count:>4} オノマトペ ({count/len(valid_counts)*100:.1f}%)')

# ===========================================
# 3. DSPパラメータの基本統計
# ===========================================
print('\n[3] DSPパラメータの基本統計（オノマトペ平均値）')
print('-' * 40)

print(f'\n{"Parameter":<15} {"Mean":>10} {"Std":>10} {"Min":>10} {"Max":>10} {"Median":>10}')
print('-' * 70)

for name in param_names:
    mean_val = onoma_dsp[name].mean()
    std_val = onoma_dsp[name].std()
    min_val = onoma_dsp[name].min()
    max_val = onoma_dsp[name].max()
    median_val = onoma_dsp[name].median()
    print(f'{name:<15} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {median_val:>10.4f}')

# ===========================================
# 4. 絶対値統計
# ===========================================
print('\n[4] DSPパラメータの絶対値統計')
print('-' * 40)

print(f'\n{"Parameter":<15} {"AbsMean":>10} {"AbsMedian":>10} {"AbsStd":>10}')
print('-' * 50)

for name in param_names:
    abs_mean = onoma_dsp[name].abs().mean()
    abs_median = onoma_dsp[name].abs().median()
    abs_std = onoma_dsp[name].abs().std()
    print(f'{name:<15} {abs_mean:>10.4f} {abs_median:>10.4f} {abs_std:>10.4f}')

# ===========================================
# 5. パーセンタイル分布
# ===========================================
print('\n[5] パーセンタイル分布')
print('-' * 40)

print(f'\n{"Parameter":<15} {"5%":>8} {"25%":>8} {"50%":>8} {"75%":>8} {"95%":>8}')
print('-' * 55)

for name in param_names:
    p5 = onoma_dsp[name].quantile(0.05)
    p25 = onoma_dsp[name].quantile(0.25)
    p50 = onoma_dsp[name].quantile(0.50)
    p75 = onoma_dsp[name].quantile(0.75)
    p95 = onoma_dsp[name].quantile(0.95)
    print(f'{name:<15} {p5:>8.4f} {p25:>8.4f} {p50:>8.4f} {p75:>8.4f} {p95:>8.4f}')

# ===========================================
# 6. 値の分布（ビン分析）
# ===========================================
print('\n[6] 値の分布')
print('-' * 40)

for name in param_names:
    print(f'\n{name}:')
    vals = onoma_dsp[name]

    # -1付近
    near_neg1 = (vals < -0.9).sum()
    # -0.5〜-0.9
    mid_neg = ((vals >= -0.9) & (vals < -0.5)).sum()
    # -0.5〜0
    low_neg = ((vals >= -0.5) & (vals < 0)).sum()
    # 0〜0.5
    low_pos = ((vals >= 0) & (vals < 0.5)).sum()
    # 0.5〜0.9
    mid_pos = ((vals >= 0.5) & (vals < 0.9)).sum()
    # 1付近
    near_pos1 = (vals >= 0.9).sum()

    total = len(vals)
    print(f'  <-0.9:     {near_neg1:>4} ({near_neg1/total*100:>5.1f}%)')
    print(f'  -0.9~-0.5: {mid_neg:>4} ({mid_neg/total*100:>5.1f}%)')
    print(f'  -0.5~0:    {low_neg:>4} ({low_neg/total*100:>5.1f}%)')
    print(f'  0~0.5:     {low_pos:>4} ({low_pos/total*100:>5.1f}%)')
    print(f'  0.5~0.9:   {mid_pos:>4} ({mid_pos/total*100:>5.1f}%)')
    print(f'  >0.9:      {near_pos1:>4} ({near_pos1/total*100:>5.1f}%)')

# ===========================================
# 7. パラメータ間の相関
# ===========================================
print('\n[7] パラメータ間の相関')
print('-' * 40)

corr = onoma_dsp.corr()

print('\n強い相関（|r| > 0.3）:')
for i in range(len(param_names)):
    for j in range(i+1, len(param_names)):
        r = corr.iloc[i, j]
        if abs(r) > 0.3:
            print(f'  {param_names[i]} vs {param_names[j]}: r = {r:.4f}')

# ===========================================
# 8. 極端な値を持つオノマトペ
# ===========================================
print('\n[8] 極端な値を持つオノマトペ')
print('-' * 40)

for name in param_names:
    # 最大値
    max_idx = onoma_dsp[name].idxmax()
    max_val = onoma_dsp[name].max()
    # 最小値
    min_idx = onoma_dsp[name].idxmin()
    min_val = onoma_dsp[name].min()

    print(f'\n{name}:')
    print(f'  最大: {max_idx} ({max_val:+.4f})')
    print(f'  最小: {min_idx} ({min_val:+.4f})')

# ===========================================
# 9. オノマトペの音韻特徴とDSPの関係
# ===========================================
print('\n[9] オノマトペの音韻特徴とDSPの関係')
print('-' * 40)

# 濁音
dakuten = ['ガ', 'ギ', 'グ', 'ゲ', 'ゴ', 'ザ', 'ジ', 'ズ', 'ゼ', 'ゾ',
           'ダ', 'ヂ', 'ヅ', 'デ', 'ド', 'バ', 'ビ', 'ブ', 'ベ', 'ボ']

def has_dakuten(s):
    return any(c in str(s) for c in dakuten)

onoma_dsp['has_dakuten'] = [has_dakuten(idx) for idx in onoma_dsp.index]

daku_count = onoma_dsp['has_dakuten'].sum()
print(f'\n濁音を含むオノマトペ: {daku_count} ({daku_count/len(onoma_dsp)*100:.1f}%)')

print(f'\nDSPパラメータの比較（濁音あり vs なし）:')
print(f'{"Parameter":<15} {"With":>10} {"Without":>10} {"Diff":>10}')
print('-' * 50)

for name in param_names:
    with_daku = onoma_dsp[onoma_dsp['has_dakuten']][name].mean()
    without_daku = onoma_dsp[~onoma_dsp['has_dakuten']][name].mean()
    diff = with_daku - without_daku
    print(f'{name:<15} {with_daku:>10.4f} {without_daku:>10.4f} {diff:>+10.4f}')

# ===========================================
# 10. オノマトペの文字数とDSPの関係
# ===========================================
print('\n[10] オノマトペの文字数とDSPの関係')
print('-' * 40)

onoma_dsp['length'] = [len(idx) for idx in onoma_dsp.index]

print(f'\n文字数の分布:')
for length in range(1, 11):
    count = (onoma_dsp['length'] == length).sum()
    if count > 0:
        print(f'  {length}文字: {count} ({count/len(onoma_dsp)*100:.1f}%)')

# 短いオノマトペ vs 長いオノマトペ
short = onoma_dsp[onoma_dsp['length'] <= 3]
long = onoma_dsp[onoma_dsp['length'] >= 6]

print(f'\nDSPパラメータの比較（短い≤3文字 vs 長い≥6文字）:')
print(f'{"Parameter":<15} {"Short":>10} {"Long":>10} {"Diff":>10}')
print('-' * 50)

for name in param_names:
    short_mean = short[name].mean()
    long_mean = long[name].mean()
    diff = long_mean - short_mean
    print(f'{name:<15} {short_mean:>10.4f} {long_mean:>10.4f} {diff:>+10.4f}')

# ===========================================
# 11. まとめ
# ===========================================
print('\n' + '=' * 80)
print('まとめ')
print('=' * 80)

# 分散ランキング
std_ranking = [(name, onoma_dsp[name].std()) for name in param_names]
std_ranking.sort(key=lambda x: x[1], reverse=True)

print(f'''
1. データ規模
   - 403種類のオノマトペ（3サンプル以上）
   - 各オノマトペの平均サンプル数: {valid_counts.mean():.1f}

2. パラメータの分散（大きい順）:''')
for rank, (name, std) in enumerate(std_ranking, 1):
    print(f'   {rank}. {name:<15} std={std:.4f}')

print(f'''
3. 値の偏り:
   - eq_sub: {(onoma_dsp['eq_sub'] < -0.9).sum()/len(onoma_dsp)*100:.1f}%が-0.9未満
   - eq_low: {(onoma_dsp['eq_low'] < -0.9).sum()/len(onoma_dsp)*100:.1f}%が-0.9未満
   - time_stretch: {(onoma_dsp['time_stretch'] > 0.9).sum()/len(onoma_dsp)*100:.1f}%が0.9以上

4. 音韻特徴との関連:
   - 濁音あり: compression +{onoma_dsp[onoma_dsp['has_dakuten']]['compression'].mean() - onoma_dsp[~onoma_dsp['has_dakuten']]['compression'].mean():.2f}, attack {onoma_dsp[onoma_dsp['has_dakuten']]['attack'].mean() - onoma_dsp[~onoma_dsp['has_dakuten']]['attack'].mean():.2f}
''')
