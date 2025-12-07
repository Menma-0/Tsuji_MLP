"""
抽出したDSPパラメータの分析
学習データとして使えるか検証する
"""
import pandas as pd
import numpy as np
import os
import sys

# 文字化け対策
sys.stdout.reconfigure(encoding='utf-8')


def main():
    print("=" * 80)
    print("抽出DSPパラメータの分析")
    print("=" * 80)

    # データ読み込み
    data_path = os.path.join(os.path.dirname(__file__), '../../data/rwcp_dataset_audio_based.csv')
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    print(f"\n総サンプル数: {len(df)}")
    print(f"ユニークなオノマトペ数: {df['onomatopoeia_katakana'].nunique()}")
    print(f"ユニークな音源数: {df['audio_path'].nunique()}")

    # DSPパラメータのカラム
    dsp_cols = [f'dsp_target_{i}' for i in range(10)]
    param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                   'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']

    # ===========================================
    # 1. 基本統計
    # ===========================================
    print("\n" + "=" * 80)
    print("1. 基本統計")
    print("=" * 80)

    print(f"\n{'Parameter':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
    print("-" * 63)

    for col, name in zip(dsp_cols, param_names):
        mean = df[col].mean()
        std = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        print(f"{name:<15} {mean:>8.3f} {std:>8.3f} {min_val:>8.3f} {max_val:>8.3f} {range_val:>8.3f}")

    # ===========================================
    # 2. 値の分布（偏り）の確認
    # ===========================================
    print("\n" + "=" * 80)
    print("2. 値の分布（偏り）")
    print("=" * 80)

    print("\n各パラメータの値域ごとの割合:")
    print(f"{'Parameter':<15} {'[-1,-0.5]':>12} {'(-0.5,0]':>12} {'(0,0.5]':>12} {'(0.5,1]':>12}")
    print("-" * 63)

    for col, name in zip(dsp_cols, param_names):
        bin1 = ((df[col] >= -1) & (df[col] <= -0.5)).sum() / len(df) * 100
        bin2 = ((df[col] > -0.5) & (df[col] <= 0)).sum() / len(df) * 100
        bin3 = ((df[col] > 0) & (df[col] <= 0.5)).sum() / len(df) * 100
        bin4 = ((df[col] > 0.5) & (df[col] <= 1)).sum() / len(df) * 100
        print(f"{name:<15} {bin1:>11.1f}% {bin2:>11.1f}% {bin3:>11.1f}% {bin4:>11.1f}%")

    # ===========================================
    # 3. オノマトペごとのDSPパラメータ分布
    # ===========================================
    print("\n" + "=" * 80)
    print("3. オノマトペごとのDSPパラメータ（上位10個）")
    print("=" * 80)

    # 頻度上位10個のオノマトペ
    top_onoma = df['onomatopoeia_katakana'].value_counts().head(10).index.tolist()

    print(f"\n{'Onomatopoeia':<12} {'Count':>6} {'gain':>8} {'eq_sub':>8} {'eq_high':>8} {'attack':>8}")
    print("-" * 50)

    for onoma in top_onoma:
        subset = df[df['onomatopoeia_katakana'] == onoma]
        count = len(subset)
        gain = subset['dsp_target_0'].mean()
        eq_sub = subset['dsp_target_2'].mean()
        eq_high = subset['dsp_target_5'].mean()
        attack = subset['dsp_target_7'].mean()
        print(f"{onoma:<12} {count:>6} {gain:>8.3f} {eq_sub:>8.3f} {eq_high:>8.3f} {attack:>8.3f}")

    # ===========================================
    # 4. 同じ音源に対する異なるオノマトペの比較
    # ===========================================
    print("\n" + "=" * 80)
    print("4. 同じ音源・異なるオノマトペの場合のDSP値")
    print("=" * 80)

    # audio_pathでグループ化して、複数のオノマトペがある音源を探す
    audio_groups = df.groupby('audio_path')['onomatopoeia_katakana'].nunique()
    multi_onoma_audios = audio_groups[audio_groups > 1].index.tolist()

    print(f"\n複数オノマトペを持つ音源数: {len(multi_onoma_audios)}")

    if len(multi_onoma_audios) > 0:
        print("\n例: 同じ音源に異なるオノマトペ")
        for audio_path in multi_onoma_audios[:3]:
            subset = df[df['audio_path'] == audio_path]
            print(f"\n  音源: {audio_path}")
            for _, row in subset.iterrows():
                onoma = row['onomatopoeia_katakana']
                gain = row['dsp_target_0']
                eq_high = row['dsp_target_5']
                attack = row['dsp_target_7']
                print(f"    {onoma}: gain={gain:.3f}, eq_high={eq_high:.3f}, attack={attack:.3f}")

    # ===========================================
    # 5. 学習データとしての問題点
    # ===========================================
    print("\n" + "=" * 80)
    print("5. 学習データとしての問題点の確認")
    print("=" * 80)

    # 問題1: 同じ音源なのにDSPが同じ
    print("\n[問題1] 同じ音源に対する異なるオノマトペでDSP値が同じか？")

    if len(multi_onoma_audios) > 0:
        same_dsp_count = 0
        diff_dsp_count = 0

        for audio_path in multi_onoma_audios:
            subset = df[df['audio_path'] == audio_path]
            dsp_values = subset[dsp_cols].values

            # 全行が同じ値かチェック
            if np.allclose(dsp_values, dsp_values[0], atol=1e-6):
                same_dsp_count += 1
            else:
                diff_dsp_count += 1

        print(f"  同じDSP値: {same_dsp_count} 音源")
        print(f"  異なるDSP値: {diff_dsp_count} 音源")

        if same_dsp_count == len(multi_onoma_audios):
            print("\n  [警告] 全ての音源でDSP値が同一！")
            print("  → DSPパラメータは音声から抽出しているため、")
            print("    同じ音声ファイルなら同じDSP値になるのは当然")
    else:
        print("  複数オノマトペを持つ音源がありません")

    # 問題2: パラメータの偏り
    print("\n[問題2] パラメータの値域の偏り")

    problematic_params = []
    for col, name in zip(dsp_cols, param_names):
        # 90%以上が特定の値域に集中している場合
        q10 = df[col].quantile(0.10)
        q90 = df[col].quantile(0.90)
        range_90 = q90 - q10

        if range_90 < 0.5:  # 80%のデータが0.5未満の範囲に集中
            problematic_params.append((name, q10, q90, range_90))

    if problematic_params:
        print("  以下のパラメータは値の分散が小さい:")
        for name, q10, q90, range_90 in problematic_params:
            print(f"    {name}: 10-90%範囲 = [{q10:.3f}, {q90:.3f}] (幅: {range_90:.3f})")
    else:
        print("  全パラメータで十分な分散があります")

    # ===========================================
    # 6. オノマトペ間の差分の分析
    # ===========================================
    print("\n" + "=" * 80)
    print("6. オノマトペ間のDSP差分（学習で使う差分の分析）")
    print("=" * 80)

    # 異なるオノマトペのペアを作成して差分を計算
    print("\n代表的なオノマトペペアのDSP差分:")

    # オノマトペごとの平均DSPを計算
    onoma_mean_dsp = df.groupby('onomatopoeia_katakana')[dsp_cols].mean()

    # いくつかのペアで差分を確認
    test_pairs = [
        ('カッ', 'ガッ'),      # 清音 vs 濁音
        ('コン', 'ドン'),      # 清音 vs 濁音
        ('カラカラ', 'ガラガラ'),  # 繰り返し清音 vs 濁音
        ('チリン', 'ゴロゴロ'),    # 高音系 vs 低音系
    ]

    print(f"\n{'Pair':<25} {'d_gain':>8} {'d_eq_sub':>9} {'d_eq_high':>10} {'d_attack':>9}")
    print("-" * 65)

    for source, target in test_pairs:
        if source in onoma_mean_dsp.index and target in onoma_mean_dsp.index:
            diff = onoma_mean_dsp.loc[target] - onoma_mean_dsp.loc[source]
            d_gain = diff['dsp_target_0']
            d_eq_sub = diff['dsp_target_2']
            d_eq_high = diff['dsp_target_5']
            d_attack = diff['dsp_target_7']
            print(f"{source}→{target:<15} {d_gain:>8.3f} {d_eq_sub:>9.3f} {d_eq_high:>10.3f} {d_attack:>9.3f}")
        else:
            print(f"{source}→{target:<15} データなし")

    # ===========================================
    # 7. 結論
    # ===========================================
    print("\n" + "=" * 80)
    print("7. 結論")
    print("=" * 80)

    print("""
このデータの構造的問題:

1. DSPパラメータは「音声ファイル」から抽出している
2. 同じ音声ファイルに複数のオノマトペが紐づいている場合、
   DSP値は全く同じになる（音声が同じなので当然）
3. つまり「オノマトペ→DSP」の対応ではなく
   「音声→DSP」の対応になっている

学習への影響:
- 「カッ」と「ガッ」が同じ音声に紐づいている場合、
  同じDSP値を持つため、清音/濁音の違いが学習できない
- オノマトペの音韻的特徴とDSPの関係が学習できない可能性

確認すべき点:
- 同じ音源に異なるオノマトペが紐づいているケースがどれくらいあるか
- そのケースでDSPの差分がゼロになってしまわないか
""")


if __name__ == '__main__':
    main()
