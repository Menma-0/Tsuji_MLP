"""
オノマトペペアとDSPパラメータ差分の学習データを作成

学習データの形式:
- source_onomatopoeia: 元のオノマトペ
- target_onomatopoeia: 目標のオノマトペ
- dsp_diff_0 ~ dsp_diff_9: DSPパラメータの差分 (target - source)
"""
import pandas as pd
import numpy as np
import os
import sys
from itertools import combinations
import random

sys.stdout.reconfigure(encoding='utf-8')


def create_training_pairs(
    input_csv: str,
    output_csv: str,
    max_pairs_per_onomatopoeia: int = 50,
    random_seed: int = 42
):
    """
    オノマトペペアの学習データを作成

    Args:
        input_csv: 音声ベースのDSPパラメータCSV
        output_csv: 出力する学習データCSV
        max_pairs_per_onomatopoeia: 1つのオノマトペあたりの最大ペア数
        random_seed: 乱数シード
    """
    print("=" * 80)
    print("学習用ペアデータの作成")
    print("=" * 80)

    random.seed(random_seed)
    np.random.seed(random_seed)

    # データ読み込み
    print(f"\n[1] データ読み込み: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    print(f"    総サンプル数: {len(df)}")
    print(f"    ユニークなオノマトペ数: {df['onomatopoeia_katakana'].nunique()}")

    # DSPパラメータのカラム
    dsp_cols = [f'dsp_target_{i}' for i in range(10)]

    # オノマトペごとの平均DSPパラメータを計算
    print("\n[2] オノマトペごとの平均DSPパラメータを計算")
    onoma_dsp = df.groupby('onomatopoeia_katakana')[dsp_cols].mean()
    print(f"    ユニークなオノマトペ数: {len(onoma_dsp)}")

    # サンプル数が少ないオノマトペを除外（信頼性のため）
    onoma_counts = df['onomatopoeia_katakana'].value_counts()
    min_samples = 3  # 最低3サンプル以上
    valid_onomas = onoma_counts[onoma_counts >= min_samples].index.tolist()
    onoma_dsp = onoma_dsp.loc[valid_onomas]
    print(f"    {min_samples}サンプル以上のオノマトペ数: {len(onoma_dsp)}")

    # ペアを作成
    print("\n[3] オノマトペペアを作成")

    all_pairs = []
    onoma_list = onoma_dsp.index.tolist()

    # 全ペアの組み合わせを生成（順序あり: A→B と B→A は別）
    for source_onoma in onoma_list:
        # このオノマトペをsourceとするペアを作成
        targets = [t for t in onoma_list if t != source_onoma]

        # ペア数が多すぎる場合はランダムサンプリング
        if len(targets) > max_pairs_per_onomatopoeia:
            targets = random.sample(targets, max_pairs_per_onomatopoeia)

        for target_onoma in targets:
            # DSP差分を計算
            source_dsp = onoma_dsp.loc[source_onoma].values
            target_dsp = onoma_dsp.loc[target_onoma].values
            dsp_diff = target_dsp - source_dsp

            pair_data = {
                'source_onomatopoeia': source_onoma,
                'target_onomatopoeia': target_onoma,
            }

            # DSP差分を追加
            for i, diff_val in enumerate(dsp_diff):
                pair_data[f'dsp_diff_{i}'] = diff_val

            all_pairs.append(pair_data)

    print(f"    生成されたペア数: {len(all_pairs)}")

    # DataFrameに変換
    pairs_df = pd.DataFrame(all_pairs)

    # 統計情報
    print("\n[4] DSP差分の統計")
    diff_cols = [f'dsp_diff_{i}' for i in range(10)]
    param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                   'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']

    print(f"\n{'Parameter':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 55)

    for col, name in zip(diff_cols, param_names):
        mean = pairs_df[col].mean()
        std = pairs_df[col].std()
        min_val = pairs_df[col].min()
        max_val = pairs_df[col].max()
        print(f"{name:<15} {mean:>10.4f} {std:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")

    # 保存
    print(f"\n[5] 保存: {output_csv}")
    pairs_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # サンプル表示
    print("\n[6] サンプルデータ（最初の10件）")
    print(pairs_df.head(10).to_string())

    print("\n" + "=" * 80)
    print(f"完了: {len(pairs_df)} ペアを {output_csv} に保存しました")
    print("=" * 80)

    return pairs_df


def create_balanced_training_pairs(
    input_csv: str,
    output_csv: str,
    n_pairs: int = 50000,
    random_seed: int = 42
):
    """
    バランスの取れた学習データを作成
    DSP差分の分布が偏らないようにサンプリング

    Args:
        input_csv: 音声ベースのDSPパラメータCSV
        output_csv: 出力する学習データCSV
        n_pairs: 生成するペア数
        random_seed: 乱数シード
    """
    print("=" * 80)
    print("バランス調整済み学習用ペアデータの作成")
    print("=" * 80)

    random.seed(random_seed)
    np.random.seed(random_seed)

    # データ読み込み
    print(f"\n[1] データ読み込み: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8-sig')

    dsp_cols = [f'dsp_target_{i}' for i in range(10)]

    # オノマトペごとの平均DSPパラメータを計算
    print("\n[2] オノマトペごとの平均DSPパラメータを計算")
    onoma_dsp = df.groupby('onomatopoeia_katakana')[dsp_cols].mean()

    # サンプル数が少ないオノマトペを除外
    onoma_counts = df['onomatopoeia_katakana'].value_counts()
    min_samples = 3
    valid_onomas = onoma_counts[onoma_counts >= min_samples].index.tolist()
    onoma_dsp = onoma_dsp.loc[valid_onomas]
    print(f"    有効なオノマトペ数: {len(onoma_dsp)}")

    # ランダムにペアを生成
    print(f"\n[3] {n_pairs}ペアをランダム生成")

    onoma_list = onoma_dsp.index.tolist()
    all_pairs = []

    for _ in range(n_pairs):
        # ランダムに2つのオノマトペを選択
        source_onoma, target_onoma = random.sample(onoma_list, 2)

        # DSP差分を計算
        source_dsp = onoma_dsp.loc[source_onoma].values
        target_dsp = onoma_dsp.loc[target_onoma].values
        dsp_diff = target_dsp - source_dsp

        pair_data = {
            'source_onomatopoeia': source_onoma,
            'target_onomatopoeia': target_onoma,
        }

        for i, diff_val in enumerate(dsp_diff):
            pair_data[f'dsp_diff_{i}'] = diff_val

        all_pairs.append(pair_data)

    pairs_df = pd.DataFrame(all_pairs)

    # 重複を除去
    pairs_df = pairs_df.drop_duplicates(subset=['source_onomatopoeia', 'target_onomatopoeia'])
    print(f"    重複除去後: {len(pairs_df)} ペア")

    # 統計情報
    print("\n[4] DSP差分の統計")
    diff_cols = [f'dsp_diff_{i}' for i in range(10)]
    param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                   'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']

    print(f"\n{'Parameter':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 55)

    for col, name in zip(diff_cols, param_names):
        mean = pairs_df[col].mean()
        std = pairs_df[col].std()
        min_val = pairs_df[col].min()
        max_val = pairs_df[col].max()
        print(f"{name:<15} {mean:>10.4f} {std:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")

    # 保存
    print(f"\n[5] 保存: {output_csv}")
    pairs_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 80)
    print(f"完了: {len(pairs_df)} ペアを {output_csv} に保存しました")
    print("=" * 80)

    return pairs_df


def main():
    """メイン関数"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    input_csv = os.path.join(project_root, 'data', 'rwcp_dataset_audio_based.csv')
    output_dir = os.path.join(project_root, 'data')

    # 方法1: 全ペア（制限付き）
    output_csv_1 = os.path.join(output_dir, 'training_pairs_full.csv')
    create_training_pairs(
        input_csv=input_csv,
        output_csv=output_csv_1,
        max_pairs_per_onomatopoeia=100,
        random_seed=42
    )

    print("\n\n")

    # 方法2: バランス調整済みランダムサンプリング
    output_csv_2 = os.path.join(output_dir, 'training_pairs_balanced.csv')
    create_balanced_training_pairs(
        input_csv=input_csv,
        output_csv=output_csv_2,
        n_pairs=100000,
        random_seed=42
    )


if __name__ == '__main__':
    main()
