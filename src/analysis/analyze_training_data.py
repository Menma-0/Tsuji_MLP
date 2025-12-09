"""
学習データの詳細分析
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(encoding='utf-8')


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    df = pd.read_csv(os.path.join(project_root, 'data', 'training_pairs_balanced.csv'))

    print('=' * 80)
    print('学習データ分析')
    print('=' * 80)

    # 基本統計
    print('\n[1] 基本情報')
    print(f'  サンプル数: {len(df)}')
    print(f'  ユニークなソースオノマトペ数: {df["source_onomatopoeia"].nunique()}')
    print(f'  ユニークなターゲットオノマトペ数: {df["target_onomatopoeia"].nunique()}')

    # 同一オノマトペペアのチェック
    same_pair = (df['source_onomatopoeia'] == df['target_onomatopoeia']).sum()
    print(f'  同一オノマトペペア数: {same_pair}')

    # DSPパラメータの差分の統計
    print('\n[2] DSPパラメータ差分の統計')
    dsp_cols = [f'dsp_diff_{i}' for i in range(10)]
    param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                   'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']

    print(f'\n{"Parameter":<15} {"Mean":>10} {"Std":>10} {"Min":>10} {"Max":>10} {"Zeros%":>10}')
    print('-' * 70)

    stats_data = []
    for i, (col, name) in enumerate(zip(dsp_cols, param_names)):
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        zero_pct = (np.abs(df[col]) < 0.001).sum() / len(df) * 100
        stats_data.append({
            'name': name, 'mean': mean_val, 'std': std_val,
            'min': min_val, 'max': max_val, 'zero_pct': zero_pct
        })
        print(f'{name:<15} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {zero_pct:>9.1f}%')

    # 符号の分布
    print('\n[3] 符号の分布（正/負/ゼロ付近）')
    print(f'\n{"Parameter":<15} {"Positive%":>12} {"Negative%":>12} {"Near Zero%":>12}')
    print('-' * 55)

    for i, (col, name) in enumerate(zip(dsp_cols, param_names)):
        pos_pct = (df[col] > 0.01).sum() / len(df) * 100
        neg_pct = (df[col] < -0.01).sum() / len(df) * 100
        zero_pct = (np.abs(df[col]) <= 0.01).sum() / len(df) * 100
        print(f'{name:<15} {pos_pct:>11.1f}% {neg_pct:>11.1f}% {zero_pct:>11.1f}%')

    # 分布の偏り（歪度・尖度）
    print('\n[4] 分布の形状（歪度・尖度）')
    print(f'\n{"Parameter":<15} {"Skewness":>12} {"Kurtosis":>12}')
    print('-' * 40)

    for i, (col, name) in enumerate(zip(dsp_cols, param_names)):
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        print(f'{name:<15} {skew:>12.4f} {kurt:>12.4f}')

    # パーセンタイル分析
    print('\n[5] パーセンタイル分布')
    print(f'\n{"Parameter":<15} {"5%":>10} {"25%":>10} {"50%":>10} {"75%":>10} {"95%":>10}')
    print('-' * 70)

    for i, (col, name) in enumerate(zip(dsp_cols, param_names)):
        p5 = df[col].quantile(0.05)
        p25 = df[col].quantile(0.25)
        p50 = df[col].quantile(0.50)
        p75 = df[col].quantile(0.75)
        p95 = df[col].quantile(0.95)
        print(f'{name:<15} {p5:>10.4f} {p25:>10.4f} {p50:>10.4f} {p75:>10.4f} {p95:>10.4f}')

    # オノマトペペアの頻度分析
    print('\n[6] オノマトペの出現頻度')
    source_counts = df['source_onomatopoeia'].value_counts()
    target_counts = df['target_onomatopoeia'].value_counts()

    print(f'\n  ソースオノマトペ:')
    print(f'    最頻出: {source_counts.index[0]} ({source_counts.iloc[0]}回)')
    print(f'    平均出現回数: {source_counts.mean():.1f}')
    print(f'    出現回数の標準偏差: {source_counts.std():.1f}')
    print(f'    最小出現回数: {source_counts.min()}')
    print(f'    最大出現回数: {source_counts.max()}')

    print(f'\n  ターゲットオノマトペ:')
    print(f'    最頻出: {target_counts.index[0]} ({target_counts.iloc[0]}回)')
    print(f'    平均出現回数: {target_counts.mean():.1f}')
    print(f'    出現回数の標準偏差: {target_counts.std():.1f}')
    print(f'    最小出現回数: {target_counts.min()}')
    print(f'    最大出現回数: {target_counts.max()}')

    # パラメータ間の相関
    print('\n[7] DSPパラメータ差分間の相関')
    dsp_df = df[dsp_cols]
    dsp_df.columns = param_names
    corr_matrix = dsp_df.corr()

    # 強い相関のペアを表示
    print('\n  強い相関（|r| > 0.3）:')
    for i in range(len(param_names)):
        for j in range(i+1, len(param_names)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.3:
                print(f'    {param_names[i]} vs {param_names[j]}: r={r:.4f}')

    # time_stretchの特別分析
    print('\n[8] time_stretchの詳細分析')
    ts_col = 'dsp_diff_9'
    ts_data = df[ts_col]

    zero_count = (ts_data == 0).sum()
    near_zero = (np.abs(ts_data) < 0.001).sum()
    print(f'  完全なゼロの数: {zero_count} ({zero_count/len(df)*100:.1f}%)')
    print(f'  ゼロ付近（|x|<0.001）の数: {near_zero} ({near_zero/len(df)*100:.1f}%)')

    # ゼロ以外の値の分布
    non_zero = ts_data[np.abs(ts_data) >= 0.001]
    if len(non_zero) > 0:
        print(f'\n  ゼロ以外の値（{len(non_zero)}サンプル）:')
        print(f'    平均: {non_zero.mean():.4f}')
        print(f'    標準偏差: {non_zero.std():.4f}')
        print(f'    最小: {non_zero.min():.4f}')
        print(f'    最大: {non_zero.max():.4f}')

    # eq_subの詳細分析
    print('\n[9] eq_sub（低域EQ）の詳細分析')
    eq_sub_col = 'dsp_diff_2'
    eq_sub_data = df[eq_sub_col]

    # 値の分布をビン化
    bins = [-np.inf, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, np.inf]
    labels = ['<-0.5', '-0.5~-0.1', '-0.1~-0.01', '-0.01~0.01', '0.01~0.1', '0.1~0.5', '>0.5']
    binned = pd.cut(eq_sub_data, bins=bins, labels=labels)

    print('\n  値の分布:')
    for label in labels:
        count = (binned == label).sum()
        pct = count / len(df) * 100
        print(f'    {label:>12}: {count:>6} ({pct:>5.1f}%)')

    # 特徴量の入力側分析
    print('\n[10] 入力特徴量（オノマトペ）の分析')

    # オノマトペの長さ分布
    source_lens = df['source_onomatopoeia'].str.len()
    target_lens = df['target_onomatopoeia'].str.len()

    print(f'\n  オノマトペの文字数:')
    print(f'    ソース: 平均{source_lens.mean():.1f}文字 (min={source_lens.min()}, max={source_lens.max()})')
    print(f'    ターゲット: 平均{target_lens.mean():.1f}文字 (min={target_lens.min()}, max={target_lens.max()})')

    # 濁音・半濁音の有無
    dakuten = ['ガ', 'ギ', 'グ', 'ゲ', 'ゴ', 'ザ', 'ジ', 'ズ', 'ゼ', 'ゾ',
               'ダ', 'ヂ', 'ヅ', 'デ', 'ド', 'バ', 'ビ', 'ブ', 'ベ', 'ボ']
    handakuten = ['パ', 'ピ', 'プ', 'ペ', 'ポ']

    def has_dakuten(s):
        return any(c in s for c in dakuten)

    def has_handakuten(s):
        return any(c in s for c in handakuten)

    source_dakuten = df['source_onomatopoeia'].apply(has_dakuten).sum()
    target_dakuten = df['target_onomatopoeia'].apply(has_dakuten).sum()
    source_handakuten = df['source_onomatopoeia'].apply(has_handakuten).sum()
    target_handakuten = df['target_onomatopoeia'].apply(has_handakuten).sum()

    print(f'\n  濁音を含むオノマトペ:')
    print(f'    ソース: {source_dakuten} ({source_dakuten/len(df)*100:.1f}%)')
    print(f'    ターゲット: {target_dakuten} ({target_dakuten/len(df)*100:.1f}%)')

    print(f'\n  半濁音を含むオノマトペ:')
    print(f'    ソース: {source_handakuten} ({source_handakuten/len(df)*100:.1f}%)')
    print(f'    ターゲット: {target_handakuten} ({target_handakuten/len(df)*100:.1f}%)')

    # まとめ
    print('\n' + '=' * 80)
    print('分析結果のまとめ')
    print('=' * 80)

    # 分散が小さいパラメータ
    low_var_params = [s['name'] for s in stats_data if s['std'] < 0.3]
    high_var_params = [s['name'] for s in stats_data if s['std'] > 0.7]
    high_zero_params = [s['name'] for s in stats_data if s['zero_pct'] > 10]

    print(f'''
1. データサイズ: {len(df)}ペア（403種類のオノマトペ）

2. 分散の問題:
   - 分散が小さいパラメータ（std < 0.3）: {', '.join(low_var_params) if low_var_params else 'なし'}
   - 分散が大きいパラメータ（std > 0.7）: {', '.join(high_var_params) if high_var_params else 'なし'}

3. ゼロ値の問題:
   - ゼロ付近が多いパラメータ（>10%）: {', '.join(high_zero_params) if high_zero_params else 'なし'}
   - time_stretchは{(df[ts_col] == 0).sum()/len(df)*100:.1f}%がゼロ

4. モデル性能との関連:
   - eq_sub（R2=0.31）: 標準偏差が小さく予測困難
   - time_stretch（符号正解率47%）: ゼロ値が多く符号予測が困難
   - eq_mid, attack（R2>0.68）: 分散が大きく学習が容易
''')


if __name__ == '__main__':
    main()
