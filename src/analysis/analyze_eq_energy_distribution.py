"""
学習データの各EQ帯域における周波数エネルギーの統計的分布を分析する
期待値との差ではなく、各帯域の生のエネルギー比率を取得
"""
import librosa
import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class EQEnergyAnalyzer:
    """各EQ帯域のエネルギーを分析するクラス"""

    def __init__(self, sr: int = 22050):
        self.sr = sr
        # EQ帯域の定義 (Hz)
        self.eq_bands = {
            'sub': (20, 80),        # 超低域
            'low': (80, 250),       # 低域
            'mid': (250, 2000),     # 中域
            'high': (2000, 6000),   # 高域
            'presence': (6000, 20000)  # 超高域
        }

    def extract_band_energies(self, audio_path: str) -> Dict[str, float]:
        """
        各EQ帯域の生のエネルギー比率を取得

        Returns:
            各帯域のエネルギー比率（全体に対する割合）
        """
        # 音声読み込み
        y, sr = librosa.load(audio_path, sr=self.sr)

        # STFT
        n_fft = 2048
        D = np.abs(librosa.stft(y, n_fft=n_fft))

        # 周波数ビン
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)

        # 各帯域のエネルギーを計算
        band_energies = {}
        total_energy = np.sum(D**2) + 1e-10

        for band_name, (low_freq, high_freq) in self.eq_bands.items():
            band_mask = (freqs >= low_freq) & (freqs < high_freq)
            band_energy = np.sum(D[band_mask]**2)
            # エネルギー比率（0〜1）
            band_energies[band_name] = band_energy / total_energy

        return band_energies


def analyze_training_data_eq_distribution():
    """学習データのEQ帯域エネルギー分布を分析"""

    print("=" * 80)
    print("学習データのEQ帯域エネルギー分布分析")
    print("=" * 80)

    # パスの設定
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    dataset_path = os.path.join(project_root, 'data', 'rwcp_dataset_audio_based.csv')
    audio_base_dir = os.path.join(project_root, 'selected_files')

    # 学習データを読み込み（フィルタリング済みのデータ）
    print(f"\n[1] 学習データを読み込み: {dataset_path}")
    df = pd.read_csv(dataset_path, encoding='utf-8-sig')
    print(f"    サンプル数: {len(df)}")

    # 分析器を初期化
    analyzer = EQEnergyAnalyzer()

    # 結果を格納
    results = {
        'sub': [],
        'low': [],
        'mid': [],
        'high': [],
        'presence': []
    }

    print(f"\n[2] 各音声ファイルのEQ帯域エネルギーを抽出中...")

    success_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        audio_path = os.path.join(audio_base_dir, row['audio_path'])

        if not os.path.exists(audio_path):
            error_count += 1
            continue

        try:
            band_energies = analyzer.extract_band_energies(audio_path)

            for band_name, energy in band_energies.items():
                results[band_name].append(energy)

            success_count += 1

            if success_count % 500 == 0:
                print(f"    処理済み: {success_count} サンプル")

        except Exception as e:
            error_count += 1
            continue

    print(f"\n    成功: {success_count}, エラー: {error_count}")

    # 統計情報を計算・表示
    print("\n" + "=" * 80)
    print("各EQ帯域のエネルギー比率の統計")
    print("=" * 80)

    print(f"\n{'Band':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10} {'25%':>10} {'75%':>10}")
    print("-" * 92)

    stats_data = []
    for band_name in ['sub', 'low', 'mid', 'high', 'presence']:
        values = np.array(results[band_name])

        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        median = np.median(values)
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)

        print(f"{band_name:<12} {mean:>10.4f} {std:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {median:>10.4f} {q25:>10.4f} {q75:>10.4f}")

        stats_data.append({
            'band': band_name,
            'freq_range': f"{analyzer.eq_bands[band_name][0]}-{analyzer.eq_bands[band_name][1]}Hz",
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'median': median,
            'q25': q25,
            'q75': q75
        })

    # エネルギー比率の合計（確認用）
    print(f"\n{'合計':<12} {sum([s['mean'] for s in stats_data]):>10.4f}")

    # 詳細なヒストグラム情報
    print("\n" + "=" * 80)
    print("各帯域のエネルギー分布（ヒストグラム）")
    print("=" * 80)

    for band_name in ['sub', 'low', 'mid', 'high', 'presence']:
        values = np.array(results[band_name])

        print(f"\n[{band_name.upper()}] ({analyzer.eq_bands[band_name][0]}-{analyzer.eq_bands[band_name][1]}Hz)")

        # 10分位でヒストグラム
        bins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
        hist, _ = np.histogram(values, bins=bins)

        for i in range(len(bins) - 1):
            pct = hist[i] / len(values) * 100
            bar = '#' * int(pct / 2)
            print(f"  {bins[i]:>5.2f}-{bins[i+1]:<5.2f}: {hist[i]:>5} ({pct:>5.1f}%) {bar}")

    # CSVに保存
    output_path = os.path.join(project_root, 'data', 'eq_energy_distribution_stats.csv')
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n統計情報を保存: {output_path}")

    # 生データも保存
    raw_output_path = os.path.join(project_root, 'data', 'eq_energy_raw_values.csv')
    raw_df = pd.DataFrame(results)
    raw_df.to_csv(raw_output_path, index=False, encoding='utf-8-sig')
    print(f"生データを保存: {raw_output_path}")

    print("\n" + "=" * 80)
    print("分析完了")
    print("=" * 80)

    return results, stats_data


if __name__ == '__main__':
    analyze_training_data_eq_distribution()
