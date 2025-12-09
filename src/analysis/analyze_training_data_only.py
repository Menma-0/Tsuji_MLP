"""
学習データのみの分析（モデル性能との関連なし）
"""
import pandas as pd
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
df = pd.read_csv(os.path.join(project_root, 'data', 'training_pairs_balanced.csv'))

print('=' * 80)
print('学習データ分析')
print('=' * 80)

# ===========================================
# 1. 基本情報
# ===========================================
print('\n[1] 基本情報')
print('-' * 40)
print(f'総ペア数: {len(df):,}')
print(f'カラム数: {len(df.columns)}')
print(f'カラム: {list(df.columns)}')

# ===========================================
# 2. オノマトペの分析
# ===========================================
print('\n[2] オノマトペの分析')
print('-' * 40)

source_unique = df['source_onomatopoeia'].nunique()
target_unique = df['target_onomatopoeia'].nunique()
all_onoma = set(df['source_onomatopoeia'].unique()) | set(df['target_onomatopoeia'].unique())

print(f'ソースオノマトペの種類数: {source_unique}')
print(f'ターゲットオノマトペの種類数: {target_unique}')
print(f'全オノマトペの種類数（重複除く）: {len(all_onoma)}')

# 同一ペアの確認
same_pair = (df['source_onomatopoeia'] == df['target_onomatopoeia']).sum()
print(f'同一オノマトペペア: {same_pair}')

# オノマトペの長さ
source_lens = df['source_onomatopoeia'].str.len()
target_lens = df['target_onomatopoeia'].str.len()
print(f'\nオノマトペの文字数:')
print(f'  ソース: 平均{source_lens.mean():.1f}, 最小{source_lens.min()}, 最大{source_lens.max()}')
print(f'  ターゲット: 平均{target_lens.mean():.1f}, 最小{target_lens.min()}, 最大{target_lens.max()}')

# 文字数分布
print(f'\n文字数の分布（ソース）:')
for length in range(1, 11):
    count = (source_lens == length).sum()
    if count > 0:
        print(f'  {length}文字: {count:,} ({count/len(df)*100:.1f}%)')

# ===========================================
# 3. オノマトペの出現頻度
# ===========================================
print('\n[3] オノマトペの出現頻度')
print('-' * 40)

source_counts = df['source_onomatopoeia'].value_counts()
target_counts = df['target_onomatopoeia'].value_counts()

print(f'ソースオノマトペ:')
print(f'  出現回数: 平均{source_counts.mean():.1f}, 標準偏差{source_counts.std():.1f}')
print(f'  最小{source_counts.min()}, 最大{source_counts.max()}')

print(f'\nターゲットオノマトペ:')
print(f'  出現回数: 平均{target_counts.mean():.1f}, 標準偏差{target_counts.std():.1f}')
print(f'  最小{target_counts.min()}, 最大{target_counts.max()}')

print(f'\nソースの最頻出オノマトペ（上位10）:')
for onoma, count in source_counts.head(10).items():
    print(f'  {onoma}: {count}回')

print(f'\nソースの最低頻度オノマトペ（下位10）:')
for onoma, count in source_counts.tail(10).items():
    print(f'  {onoma}: {count}回')

# ===========================================
# 4. DSPパラメータ差分の基本統計
# ===========================================
print('\n[4] DSPパラメータ差分の基本統計')
print('-' * 40)

dsp_cols = [f'dsp_diff_{i}' for i in range(10)]
param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
               'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']

print(f'\n{"Parameter":<15} {"Mean":>10} {"Std":>10} {"Min":>10} {"Max":>10} {"Median":>10}')
print('-' * 70)

for col, name in zip(dsp_cols, param_names):
    print(f'{name:<15} {df[col].mean():>10.4f} {df[col].std():>10.4f} '
          f'{df[col].min():>10.4f} {df[col].max():>10.4f} {df[col].median():>10.4f}')

# ===========================================
# 5. DSPパラメータ差分の分布特性
# ===========================================
print('\n[5] DSPパラメータ差分の分布特性')
print('-' * 40)

print(f'\n{"Parameter":<15} {"Skewness":>10} {"Kurtosis":>10} {"IQR":>10}')
print('-' * 50)

for col, name in zip(dsp_cols, param_names):
    skew = df[col].skew()
    kurt = df[col].kurtosis()
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    print(f'{name:<15} {skew:>10.4f} {kurt:>10.4f} {iqr:>10.4f}')

# ===========================================
# 6. DSPパラメータの符号分布
# ===========================================
print('\n[6] DSPパラメータの符号分布')
print('-' * 40)

print(f'\n{"Parameter":<15} {"Positive":>12} {"Negative":>12} {"Zero(±0.01)":>12}')
print('-' * 55)

for col, name in zip(dsp_cols, param_names):
    pos = (df[col] > 0.01).sum()
    neg = (df[col] < -0.01).sum()
    zero = ((df[col] >= -0.01) & (df[col] <= 0.01)).sum()
    print(f'{name:<15} {pos:>10} ({pos/len(df)*100:>4.1f}%) '
          f'{neg:>10} ({neg/len(df)*100:>4.1f}%) '
          f'{zero:>10} ({zero/len(df)*100:>4.1f}%)')

# ===========================================
# 7. パーセンタイル分布
# ===========================================
print('\n[7] パーセンタイル分布')
print('-' * 40)

print(f'\n{"Parameter":<15} {"1%":>8} {"5%":>8} {"25%":>8} {"50%":>8} {"75%":>8} {"95%":>8} {"99%":>8}')
print('-' * 80)

for col, name in zip(dsp_cols, param_names):
    p1 = df[col].quantile(0.01)
    p5 = df[col].quantile(0.05)
    p25 = df[col].quantile(0.25)
    p50 = df[col].quantile(0.50)
    p75 = df[col].quantile(0.75)
    p95 = df[col].quantile(0.95)
    p99 = df[col].quantile(0.99)
    print(f'{name:<15} {p1:>8.3f} {p5:>8.3f} {p25:>8.3f} {p50:>8.3f} {p75:>8.3f} {p95:>8.3f} {p99:>8.3f}')

# ===========================================
# 8. DSPパラメータ間の相関
# ===========================================
print('\n[8] DSPパラメータ間の相関')
print('-' * 40)

dsp_df = df[dsp_cols].copy()
dsp_df.columns = param_names
corr = dsp_df.corr()

print('\n相関行列:')
print(f'{"":>15}', end='')
for name in param_names[:5]:
    print(f'{name[:6]:>8}', end='')
print()

for i, name in enumerate(param_names[:5]):
    print(f'{name:<15}', end='')
    for j in range(5):
        print(f'{corr.iloc[i, j]:>8.3f}', end='')
    print()

print(f'\n{"":>15}', end='')
for name in param_names[5:]:
    print(f'{name[:6]:>8}', end='')
print()

for i, name in enumerate(param_names[5:], 5):
    print(f'{name:<15}', end='')
    for j in range(5, 10):
        print(f'{corr.iloc[i, j]:>8.3f}', end='')
    print()

# 強い相関のペア
print('\n強い相関（|r| > 0.3）:')
for i in range(len(param_names)):
    for j in range(i+1, len(param_names)):
        r = corr.iloc[i, j]
        if abs(r) > 0.3:
            print(f'  {param_names[i]} vs {param_names[j]}: r = {r:.4f}')

# ===========================================
# 9. オノマトペの音韻特徴
# ===========================================
print('\n[9] オノマトペの音韻特徴')
print('-' * 40)

# 濁音
dakuten = ['ガ', 'ギ', 'グ', 'ゲ', 'ゴ', 'ザ', 'ジ', 'ズ', 'ゼ', 'ゾ',
           'ダ', 'ヂ', 'ヅ', 'デ', 'ド', 'バ', 'ビ', 'ブ', 'ベ', 'ボ']
# 半濁音
handakuten = ['パ', 'ピ', 'プ', 'ペ', 'ポ']
# 促音
sokuon = ['ッ']
# 長音
chouon = ['ー']
# 撥音
hatsuon = ['ン']

def count_char_type(series, chars):
    return series.apply(lambda s: any(c in s for c in chars)).sum()

print('ソースオノマトペの特徴:')
print(f'  濁音を含む: {count_char_type(df["source_onomatopoeia"], dakuten):,} ({count_char_type(df["source_onomatopoeia"], dakuten)/len(df)*100:.1f}%)')
print(f'  半濁音を含む: {count_char_type(df["source_onomatopoeia"], handakuten):,} ({count_char_type(df["source_onomatopoeia"], handakuten)/len(df)*100:.1f}%)')
print(f'  促音を含む: {count_char_type(df["source_onomatopoeia"], sokuon):,} ({count_char_type(df["source_onomatopoeia"], sokuon)/len(df)*100:.1f}%)')
print(f'  長音を含む: {count_char_type(df["source_onomatopoeia"], chouon):,} ({count_char_type(df["source_onomatopoeia"], chouon)/len(df)*100:.1f}%)')
print(f'  撥音を含む: {count_char_type(df["source_onomatopoeia"], hatsuon):,} ({count_char_type(df["source_onomatopoeia"], hatsuon)/len(df)*100:.1f}%)')

print('\nターゲットオノマトペの特徴:')
print(f'  濁音を含む: {count_char_type(df["target_onomatopoeia"], dakuten):,} ({count_char_type(df["target_onomatopoeia"], dakuten)/len(df)*100:.1f}%)')
print(f'  半濁音を含む: {count_char_type(df["target_onomatopoeia"], handakuten):,} ({count_char_type(df["target_onomatopoeia"], handakuten)/len(df)*100:.1f}%)')
print(f'  促音を含む: {count_char_type(df["target_onomatopoeia"], sokuon):,} ({count_char_type(df["target_onomatopoeia"], sokuon)/len(df)*100:.1f}%)')
print(f'  長音を含む: {count_char_type(df["target_onomatopoeia"], chouon):,} ({count_char_type(df["target_onomatopoeia"], chouon)/len(df)*100:.1f}%)')
print(f'  撥音を含む: {count_char_type(df["target_onomatopoeia"], hatsuon):,} ({count_char_type(df["target_onomatopoeia"], hatsuon)/len(df)*100:.1f}%)')

# ===========================================
# 10. 先頭文字の分析
# ===========================================
print('\n[10] 先頭文字の分析')
print('-' * 40)

source_first = df['source_onomatopoeia'].str[0]
target_first = df['target_onomatopoeia'].str[0]

source_first_counts = source_first.value_counts()
target_first_counts = target_first.value_counts()

print('\nソースの先頭文字（上位15）:')
for char, count in source_first_counts.head(15).items():
    print(f'  {char}: {count:,} ({count/len(df)*100:.1f}%)')

# ===========================================
# 11. 繰り返しパターン
# ===========================================
print('\n[11] 繰り返しパターンの分析')
print('-' * 40)

def is_repeat(s):
    """ABAB形式かどうか"""
    if len(s) == 4:
        return s[0:2] == s[2:4]
    elif len(s) == 6:
        return s[0:3] == s[3:6]
    return False

def is_reduplication(s):
    """同じ文字の連続があるか"""
    for i in range(len(s) - 1):
        if s[i] == s[i+1]:
            return True
    return False

source_repeat = df['source_onomatopoeia'].apply(is_repeat).sum()
target_repeat = df['target_onomatopoeia'].apply(is_repeat).sum()

print(f'ABAB型の繰り返し:')
print(f'  ソース: {source_repeat:,} ({source_repeat/len(df)*100:.1f}%)')
print(f'  ターゲット: {target_repeat:,} ({target_repeat/len(df)*100:.1f}%)')

source_redup = df['source_onomatopoeia'].apply(is_reduplication).sum()
target_redup = df['target_onomatopoeia'].apply(is_reduplication).sum()

print(f'\n同じ文字の連続:')
print(f'  ソース: {source_redup:,} ({source_redup/len(df)*100:.1f}%)')
print(f'  ターゲット: {target_redup:,} ({target_redup/len(df)*100:.1f}%)')

# ===========================================
# 12. DSP差分の極端な値
# ===========================================
print('\n[12] DSP差分の極端な値')
print('-' * 40)

print('\n各パラメータで差分が大きいペア（上位3）:')
for col, name in zip(dsp_cols, param_names):
    print(f'\n{name}:')
    # 正方向
    top_pos = df.nlargest(3, col)[['source_onomatopoeia', 'target_onomatopoeia', col]]
    for _, row in top_pos.iterrows():
        print(f'  {row["source_onomatopoeia"]} → {row["target_onomatopoeia"]}: {row[col]:+.4f}')
    # 負方向
    top_neg = df.nsmallest(3, col)[['source_onomatopoeia', 'target_onomatopoeia', col]]
    for _, row in top_neg.iterrows():
        print(f'  {row["source_onomatopoeia"]} → {row["target_onomatopoeia"]}: {row[col]:+.4f}')

# ===========================================
# 13. データの品質チェック
# ===========================================
print('\n[13] データの品質チェック')
print('-' * 40)

# 欠損値
print(f'\n欠損値:')
for col in df.columns:
    na_count = df[col].isna().sum()
    if na_count > 0:
        print(f'  {col}: {na_count}')
if df.isna().sum().sum() == 0:
    print('  なし')

# 重複行
dup_count = df.duplicated().sum()
print(f'\n完全な重複行: {dup_count}')

# ペアの重複（source, target組み合わせ）
pair_dup = df.duplicated(subset=['source_onomatopoeia', 'target_onomatopoeia']).sum()
print(f'オノマトペペアの重複: {pair_dup}')

# 逆ペアの確認
df['pair'] = df['source_onomatopoeia'] + '_' + df['target_onomatopoeia']
df['reverse_pair'] = df['target_onomatopoeia'] + '_' + df['source_onomatopoeia']
reverse_exists = df['reverse_pair'].isin(df['pair']).sum()
print(f'逆方向ペアも存在する数: {reverse_exists:,}')

# ===========================================
# 14. まとめ
# ===========================================
print('\n' + '=' * 80)
print('まとめ')
print('=' * 80)

print(f'''
1. データ規模
   - 総ペア数: {len(df):,}
   - オノマトペ種類: {len(all_onoma)}種類
   - 各オノマトペの出現回数: 約{source_counts.mean():.0f}回（バランス良好）

2. オノマトペの特徴
   - 平均文字数: {source_lens.mean():.1f}文字
   - 濁音含有率: ソース{count_char_type(df["source_onomatopoeia"], dakuten)/len(df)*100:.1f}%, ターゲット{count_char_type(df["target_onomatopoeia"], dakuten)/len(df)*100:.1f}%
   - 促音含有率: ソース{count_char_type(df["source_onomatopoeia"], sokuon)/len(df)*100:.1f}%, ターゲット{count_char_type(df["target_onomatopoeia"], sokuon)/len(df)*100:.1f}%

3. DSPパラメータ差分
   - 全パラメータで平均≒0（ペア作成時のバランシングが機能）
   - 標準偏差が大きい: eq_mid({df["dsp_diff_4"].std():.2f}), eq_presence({df["dsp_diff_6"].std():.2f}), attack({df["dsp_diff_7"].std():.2f})
   - 標準偏差が小さい: eq_sub({df["dsp_diff_2"].std():.2f})

4. パラメータ間の相関
   - 強い正の相関: eq_sub vs eq_low (r=0.43)
   - 強い負の相関: eq_mid vs eq_presence (r=-0.60), attack vs sustain (r=-0.58)

5. データ品質
   - 欠損値: なし
   - 重複行: {dup_count}
   - 逆方向ペアも存在: {reverse_exists:,}ペア（学習に有効）
''')
