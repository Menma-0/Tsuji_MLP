"""
元データ（音声から抽出したDSP）の分析
"""
import pandas as pd
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
df = pd.read_csv(os.path.join(project_root, 'data', 'rwcp_dataset_audio_based.csv'))

# カラム名を修正
df.rename(columns={'onomatopoeia_katakana': 'onomatopoeia'}, inplace=True)

print('=' * 80)
print('元データ（音声から抽出したDSP）の分析')
print('=' * 80)

print(f'\nサンプル数: {len(df)}')
print(f'ユニークなオノマトペ数: {df["onomatopoeia"].nunique()}')

# DSPパラメータの統計（カラム名がdsp_target_Nになっている）
dsp_cols_orig = [f'dsp_target_{i}' for i in range(10)]
dsp_cols = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
            'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']
# カラム名をリネーム
for old, new in zip(dsp_cols_orig, dsp_cols):
    if old in df.columns:
        df.rename(columns={old: new}, inplace=True)

print(f'\nDSPパラメータの統計（元データ）:')
print(f'{"Parameter":<15} {"Mean":>10} {"Std":>10} {"Min":>10} {"Max":>10}')
print('-' * 60)

for col in dsp_cols:
    if col in df.columns:
        print(f'{col:<15} {df[col].mean():>10.4f} {df[col].std():>10.4f} {df[col].min():>10.4f} {df[col].max():>10.4f}')

# 特にeq_subとtime_stretchを詳しく見る
print('\n' + '=' * 80)
print('問題パラメータの詳細分析')
print('=' * 80)

# eq_sub
print('\n[1] eq_sub')
eq_sub = df['eq_sub']
print(f'  -1.0の値の数: {(eq_sub == -1.0).sum()} ({(eq_sub == -1.0).sum()/len(df)*100:.1f}%)')
print(f'  -1.0付近（< -0.95）の数: {(eq_sub < -0.95).sum()} ({(eq_sub < -0.95).sum()/len(df)*100:.1f}%)')
print(f'  分布の範囲: {eq_sub.min():.4f} ~ {eq_sub.max():.4f}')

# パーセンタイル
print(f'  パーセンタイル:')
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f'    {p}%: {eq_sub.quantile(p/100):.4f}')

# eq_low
print('\n[2] eq_low')
eq_low = df['eq_low']
print(f'  -1.0の値の数: {(eq_low == -1.0).sum()} ({(eq_low == -1.0).sum()/len(df)*100:.1f}%)')
print(f'  -1.0付近（< -0.95）の数: {(eq_low < -0.95).sum()} ({(eq_low < -0.95).sum()/len(df)*100:.1f}%)')
print(f'  分布の範囲: {eq_low.min():.4f} ~ {eq_low.max():.4f}')

print(f'  パーセンタイル:')
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f'    {p}%: {eq_low.quantile(p/100):.4f}')

# time_stretch
print('\n[3] time_stretch')
ts = df['time_stretch']
print(f'  0.0の値の数: {(ts == 0.0).sum()} ({(ts == 0.0).sum()/len(df)*100:.1f}%)')
unique_vals = ts.nunique()
print(f'  ユニークな値の数: {unique_vals}')
print(f'  分布の範囲: {ts.min():.4f} ~ {ts.max():.4f}')

# 最頻値を表示
top_vals = ts.value_counts().head(10)
print(f'  最頻値（上位10）:')
for val, count in top_vals.items():
    print(f'    {val:.6f}: {count}回 ({count/len(df)*100:.2f}%)')

# オノマトペごとのDSP値のばらつき
print('\n' + '=' * 80)
print('オノマトペごとのパラメータのばらつき')
print('=' * 80)

# 同じオノマトペで複数の音声がある場合のばらつきを見る
onoma_counts = df['onomatopoeia'].value_counts()
multi_onoma = onoma_counts[onoma_counts > 1].index

print(f'\n複数の音声を持つオノマトペ数: {len(multi_onoma)}')

if len(multi_onoma) > 0:
    print(f'\nパラメータ内分散（同一オノマトペ内）vs パラメータ間分散（全体）:')
    print(f'{"Parameter":<15} {"Within-Var":>12} {"Total-Var":>12} {"Ratio":>10}')
    print('-' * 55)

    for col in dsp_cols:
        if col in df.columns:
            # 全体の分散
            total_var = df[col].var()

            # オノマトペ内分散の平均
            within_vars = df.groupby('onomatopoeia')[col].var()
            within_var = within_vars.mean()

            ratio = within_var / total_var if total_var > 0 else 0
            print(f'{col:<15} {within_var:>12.4f} {total_var:>12.4f} {ratio:>10.2%}')

# オノマトペの特徴分析
print('\n' + '=' * 80)
print('濁音・清音とDSPパラメータの関係')
print('=' * 80)

# 濁音を含むかどうかでグループ分け
dakuten = ['ガ', 'ギ', 'グ', 'ゲ', 'ゴ', 'ザ', 'ジ', 'ズ', 'ゼ', 'ゾ',
           'ダ', 'ヂ', 'ヅ', 'デ', 'ド', 'バ', 'ビ', 'ブ', 'ベ', 'ボ']

def has_dakuten(s):
    return any(c in str(s) for c in dakuten)

df['has_dakuten'] = df['onomatopoeia'].apply(has_dakuten)

print(f'\n濁音を含むサンプル数: {df["has_dakuten"].sum()} ({df["has_dakuten"].sum()/len(df)*100:.1f}%)')

# 濁音の有無でDSPパラメータを比較
print(f'\nDSPパラメータの比較（濁音あり vs なし）:')
print(f'{"Parameter":<15} {"With Daku":>12} {"Without":>12} {"Diff":>10}')
print('-' * 55)

for col in dsp_cols:
    if col in df.columns:
        with_daku = df[df['has_dakuten']][col].mean()
        without_daku = df[~df['has_dakuten']][col].mean()
        diff = with_daku - without_daku
        print(f'{col:<15} {with_daku:>12.4f} {without_daku:>12.4f} {diff:>+10.4f}')

# まとめ
print('\n' + '=' * 80)
print('まとめ')
print('=' * 80)

print('''
1. eq_subの問題:
   - 元データで既に多くの値が-1.0付近に集中
   - 音声の低域（サブベース帯域）のエネルギーが少ない
   - ペアの差分を取っても分散が小さいまま
   → 元音声データの特性による制約

2. time_stretchの問題:
   - 元データのtime_stretch計算方法に問題がある可能性
   - 多くの値が特定の値に集中している
   → 抽出アルゴリズムの見直しが必要か

3. 濁音とDSPの関係:
   - 濁音を含むオノマトペは低域（eq_sub, eq_low）が高い傾向
   - これは音響的に妥当（濁音は低周波成分を含む）
   - モデルがこの関係を学習できていれば良好

4. モデル性能への影響:
   - eq_sub: 元データの分散が小さいため予測精度が低い
   - time_stretch: データの偏りで符号予測が困難
   - eq_mid, attack: 分散が大きく学習が容易
''')
