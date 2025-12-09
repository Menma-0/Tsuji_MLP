"""
音声ファイルから音響特徴量を抽出し、DSPパラメータを推定する

音響特徴量からDSPパラメータへのマッピング根拠:
- 音量(Gain): RMSエネルギーから推定
- コンプレッション(Compression): ダイナミックレンジから推定
- EQ帯域: スペクトル解析による各周波数帯域のエネルギー比率から推定
- トランジェント(Attack/Sustain): エンベロープ解析から推定
- タイムストレッチ: 音声の持続時間から推定
"""
import librosa
import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class AudioDSPExtractor:
    """音声ファイルからDSPパラメータを抽出するクラス"""

    def __init__(self, sr: int = 22050):
        """
        Args:
            sr: サンプリングレート
        """
        self.sr = sr

        # EQ帯域の定義 (Hz)
        self.eq_bands = {
            'sub': (20, 80),        # 超低域
            'low': (80, 250),       # 低域
            'mid': (250, 2000),     # 中域
            'high': (2000, 6000),   # 高域
            'presence': (6000, 20000)  # 超高域
        }

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        音声ファイルを読み込む

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            (音声信号, サンプリングレート)
        """
        y, sr = librosa.load(audio_path, sr=self.sr)
        return y, sr

    def extract_gain(self, y: np.ndarray) -> float:
        """
        音量(Gain)を推定
        RMSエネルギーを-1〜+1に正規化

        Args:
            y: 音声信号

        Returns:
            gain値 (-1〜+1)
        """
        rms = np.sqrt(np.mean(y**2))

        # RMSの典型的な範囲(0.001〜0.3)を-1〜+1にマッピング
        # log scaleで変換
        if rms < 1e-6:
            return -1.0

        rms_db = 20 * np.log10(rms)
        # -60dB〜0dBを-1〜+1にマッピング
        gain = np.clip((rms_db + 30) / 30, -1, 1)

        return float(gain)

    def extract_compression(self, y: np.ndarray) -> float:
        """
        コンプレッション(ダイナミクス)を推定
        ダイナミックレンジが小さいほど圧縮されている

        Args:
            y: 音声信号

        Returns:
            compression値 (-1〜+1)
        """
        # フレームごとのRMSを計算
        frame_length = 2048
        hop_length = 512

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        if len(rms) < 2:
            return 0.0

        # ダイナミックレンジを計算 (上位10%と下位10%の差)
        rms_sorted = np.sort(rms[rms > 1e-6])  # 無音部分を除外

        if len(rms_sorted) < 2:
            return 0.0

        high_idx = int(len(rms_sorted) * 0.9)
        low_idx = int(len(rms_sorted) * 0.1)

        if high_idx <= low_idx:
            return 0.0

        dynamic_range_db = 20 * np.log10(rms_sorted[high_idx] / max(rms_sorted[low_idx], 1e-6))

        # ダイナミックレンジが小さい(0-10dB)=高圧縮(+1)、大きい(30dB以上)=低圧縮(-1)
        compression = np.clip(1.0 - (dynamic_range_db / 30), -1, 1)

        return float(compression)

    def extract_eq_bands(self, y: np.ndarray) -> Dict[str, float]:
        """
        各EQ帯域のエネルギー比率を推定

        Args:
            y: 音声信号

        Returns:
            各帯域のEQ値 (-1〜+1)
        """
        # STFT
        n_fft = 2048
        D = np.abs(librosa.stft(y, n_fft=n_fft))

        # 周波数ビン
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)

        # 各帯域のエネルギーを計算
        band_energies = {}
        total_energy = np.sum(D**2) + 1e-10

        for band_name, (low_freq, high_freq) in self.eq_bands.items():
            # 該当する周波数ビンを取得
            band_mask = (freqs >= low_freq) & (freqs < high_freq)
            band_energy = np.sum(D[band_mask]**2)
            band_energies[band_name] = band_energy / total_energy

        # 各帯域の期待される比率（RWCPデータセットの平均値に基づく）
        # analyze_eq_energy_distribution.py で計算した8542サンプルの平均
        expected_ratios = {
            'sub': 0.0041,       # 20-80Hz: 平均 0.41%
            'low': 0.0206,       # 80-250Hz: 平均 2.06%
            'mid': 0.3020,       # 250-2000Hz: 平均 30.20%
            'high': 0.4555,      # 2000-6000Hz: 平均 45.55%
            'presence': 0.2176   # 6000-20000Hz: 平均 21.76%
        }

        # 期待値との差を-1〜+1にマッピング
        eq_values = {}
        for band_name in self.eq_bands.keys():
            actual = band_energies[band_name]
            expected = expected_ratios[band_name]

            # 比率の差をスケーリング（2倍以上/以下で±1）
            if expected > 0:
                ratio = actual / expected
                eq_value = np.clip((ratio - 1.0) / 1.0, -1, 1)
            else:
                eq_value = 0.0

            eq_values[band_name] = float(eq_value)

        return eq_values

    def extract_transient(self, y: np.ndarray) -> Tuple[float, float]:
        """
        トランジェント特性（アタック/サスティン）を推定

        Args:
            y: 音声信号

        Returns:
            (attack値, sustain値) 各-1〜+1
        """
        # エンベロープを計算
        envelope = np.abs(librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0])

        if len(envelope) < 3:
            return 0.0, 0.0

        # ピーク位置を見つける
        peak_idx = np.argmax(envelope)
        peak_value = envelope[peak_idx]

        if peak_value < 1e-6:
            return 0.0, 0.0

        # アタック時間: 10%から90%に達するまでの時間
        threshold_low = 0.1 * peak_value
        threshold_high = 0.9 * peak_value

        attack_start = 0
        attack_end = peak_idx

        for i in range(peak_idx):
            if envelope[i] >= threshold_low and attack_start == 0:
                attack_start = i
            if envelope[i] >= threshold_high:
                attack_end = i
                break

        attack_samples = (attack_end - attack_start) * 128  # hop_length
        attack_time_ms = (attack_samples / self.sr) * 1000

        # アタック時間を-1〜+1にマッピング
        # 0-5ms: 非常に速い(+1), 50ms以上: 遅い(-1)
        attack_value = np.clip(1.0 - (attack_time_ms / 25), -1, 1)

        # サスティン: ピーク後に50%以上を維持している時間
        sustain_threshold = 0.5 * peak_value
        sustain_samples = 0

        for i in range(peak_idx, len(envelope)):
            if envelope[i] >= sustain_threshold:
                sustain_samples += 128
            else:
                break

        sustain_time_ms = (sustain_samples / self.sr) * 1000

        # サスティン時間を-1〜+1にマッピング
        # 0-10ms: 短い(-1), 200ms以上: 長い(+1)
        sustain_value = np.clip((sustain_time_ms - 50) / 100, -1, 1)

        return float(attack_value), float(sustain_value)

    def extract_time_stretch(self, y: np.ndarray) -> float:
        """
        タイムストレッチ特性を推定
        音声の持続時間に基づく

        Args:
            y: 音声信号

        Returns:
            time_stretch値 (-1〜+1)
        """
        # 有効な音声の長さを計算（-40dB以上の部分）
        frame_length = 2048
        hop_length = 512

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # -40dB以上のフレームをカウント
        threshold = np.max(rms) * 0.01  # -40dB
        active_frames = np.sum(rms > threshold)

        duration_ms = (active_frames * hop_length / self.sr) * 1000

        # 持続時間を-1〜+1にマッピング
        # 50ms以下: 短い(-1), 500ms以上: 長い(+1)
        time_stretch = np.clip((duration_ms - 200) / 200, -1, 1)

        return float(time_stretch)

    def extract_dsp_params(self, audio_path: str) -> Dict[str, float]:
        """
        音声ファイルからDSPパラメータを抽出

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            DSPパラメータの辞書
        """
        # 音声読み込み
        y, sr = self.load_audio(audio_path)

        # 各パラメータを抽出
        gain = self.extract_gain(y)
        compression = self.extract_compression(y)
        eq_bands = self.extract_eq_bands(y)
        attack, sustain = self.extract_transient(y)
        time_stretch = self.extract_time_stretch(y)

        return {
            'gain': gain,
            'compression': compression,
            'eq_sub': eq_bands['sub'],
            'eq_low': eq_bands['low'],
            'eq_mid': eq_bands['mid'],
            'eq_high': eq_bands['high'],
            'eq_presence': eq_bands['presence'],
            'transient_attack': attack,
            'transient_sustain': sustain,
            'time_stretch': time_stretch
        }


def process_rwcp_dataset(csv_path: str, audio_base_dir: str, output_path: str):
    """
    RWCPデータセット全体を処理してDSPパラメータを抽出

    Args:
        csv_path: training_data_jp_utf8bom.csv のパス
        audio_base_dir: selected_files ディレクトリのパス
        output_path: 出力CSVのパス
    """
    print("=" * 80)
    print("RWCP音声データからDSPパラメータを抽出")
    print("=" * 80)

    # CSVを読み込み
    print(f"\n[1] CSVを読み込み: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"    総サンプル数: {len(df)}")

    # 高品質サンプルのみフィルタ
    print("\n[2] 高品質サンプルをフィルタ (confidence >= 4, avg_acceptability >= 4)")
    df_filtered = df[(df['confidence'] >= 4) & (df['avg_acceptability'] >= 4.0)]
    print(f"    フィルタ後: {len(df_filtered)} サンプル")

    # 抽出器を初期化
    extractor = AudioDSPExtractor()

    # 結果を格納するリスト
    results = []

    print("\n[3] DSPパラメータを抽出中...")

    success_count = 0
    error_count = 0

    for idx, row in df_filtered.iterrows():
        # audio_pathを変換
        original_path = row['audio_path']
        # RWCP-SSD/drysrc/a1/cherry1/043.wav → selected_files/a1/cherry1/043.wav
        relative_path = original_path.replace('RWCP-SSD/drysrc/', '')
        audio_path = os.path.join(audio_base_dir, relative_path)

        # sound_idを生成
        sound_id = relative_path.replace('/', '_').replace('.wav', '')

        if not os.path.exists(audio_path):
            error_count += 1
            continue

        try:
            # DSPパラメータを抽出
            dsp_params = extractor.extract_dsp_params(audio_path)

            # 結果を追加
            result = {
                'sound_id': sound_id,
                'audio_path': relative_path,
                'onomatopoeia_katakana': row['onomatopoeia'],
                'confidence_score': row['confidence'],
                'acceptance_score': row['avg_acceptability'],
                'dsp_target_0': dsp_params['gain'],
                'dsp_target_1': dsp_params['compression'],
                'dsp_target_2': dsp_params['eq_sub'],
                'dsp_target_3': dsp_params['eq_low'],
                'dsp_target_4': dsp_params['eq_mid'],
                'dsp_target_5': dsp_params['eq_high'],
                'dsp_target_6': dsp_params['eq_presence'],
                'dsp_target_7': dsp_params['transient_attack'],
                'dsp_target_8': dsp_params['transient_sustain'],
                'dsp_target_9': dsp_params['time_stretch']
            }
            results.append(result)
            success_count += 1

            if success_count % 500 == 0:
                print(f"    処理済み: {success_count} サンプル")

        except Exception as e:
            error_count += 1
            continue

    print(f"\n    成功: {success_count}, エラー: {error_count}")

    # DataFrameに変換
    result_df = pd.DataFrame(results)

    # 保存
    print(f"\n[4] 結果を保存: {output_path}")
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    # 統計情報を表示
    print("\n" + "=" * 80)
    print("抽出結果の統計")
    print("=" * 80)

    dsp_cols = [f'dsp_target_{i}' for i in range(10)]
    param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                   'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']

    print(f"\n{'Parameter':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)

    for i, (col, name) in enumerate(zip(dsp_cols, param_names)):
        mean = result_df[col].mean()
        std = result_df[col].std()
        min_val = result_df[col].min()
        max_val = result_df[col].max()
        print(f"{name:<20} {mean:>10.3f} {std:>10.3f} {min_val:>10.3f} {max_val:>10.3f}")

    print("\n" + "=" * 80)
    print(f"完了: {len(result_df)} サンプルを {output_path} に保存しました")
    print("=" * 80)

    return result_df


def main():
    """メイン関数"""
    # パスの設定
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    csv_path = os.path.join(project_root, 'training_data_jp_utf8bom.csv')
    audio_base_dir = os.path.join(project_root, 'selected_files')
    output_path = os.path.join(project_root, 'data', 'rwcp_dataset_audio_based.csv')

    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 処理を実行
    process_rwcp_dataset(csv_path, audio_base_dir, output_path)


if __name__ == '__main__':
    main()
