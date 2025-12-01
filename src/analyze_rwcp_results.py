"""
RWCP モデルでの音声処理結果を分析
"""
import numpy as np
import librosa
import os
import glob


def analyze_audio(filepath):
    """音声ファイルを分析"""
    audio, sr = librosa.load(filepath, sr=None)

    # 基本統計
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))

    # スペクトル分析
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    return {
        'duration': duration,
        'rms': rms,
        'peak': peak,
        'spectral_centroid': spectral_centroid
    }


def main():
    print("=" * 80)
    print("RWCP MODEL PROCESSING RESULTS ANALYSIS")
    print("=" * 80)

    output_dir = '../demo_audio/rwcp_test'

    # ペアファイルを取得
    original_files = sorted(glob.glob(os.path.join(output_dir, '*_original.wav')))

    print(f"\nFound {len(original_files)} original files\n")

    for orig_file in original_files:
        # 対応する処理済みファイルを見つける
        base_name = os.path.basename(orig_file).replace('_original.wav', '')
        processed_files = glob.glob(os.path.join(output_dir, f'{base_name}_*.wav'))
        processed_files = [f for f in processed_files if not f.endswith('_original.wav')]

        if not processed_files:
            continue

        processed_file = processed_files[0]

        # 分析
        orig_stats = analyze_audio(orig_file)
        proc_stats = analyze_audio(processed_file)

        # オノマトペを抽出
        onoma = os.path.basename(processed_file).replace('.wav', '').split('_')[-1]

        # 変化率を計算
        rms_change = ((proc_stats['rms'] - orig_stats['rms']) / orig_stats['rms']) * 100
        peak_change = ((proc_stats['peak'] - orig_stats['peak']) / orig_stats['peak']) * 100
        centroid_change = ((proc_stats['spectral_centroid'] - orig_stats['spectral_centroid']) / orig_stats['spectral_centroid']) * 100

        print("=" * 80)
        print(f"File: {base_name}")
        print(f"Onomatopoeia: {onoma}")
        print("=" * 80)

        print(f"\n[Original]")
        print(f"  Duration:           {orig_stats['duration']:.3f}s")
        print(f"  RMS:                {orig_stats['rms']:.4f}")
        print(f"  Peak:               {orig_stats['peak']:.4f}")
        print(f"  Spectral Centroid:  {orig_stats['spectral_centroid']:.1f} Hz")

        print(f"\n[Processed]")
        print(f"  Duration:           {proc_stats['duration']:.3f}s")
        print(f"  RMS:                {proc_stats['rms']:.4f} ({rms_change:+.1f}%)")
        print(f"  Peak:               {proc_stats['peak']:.4f} ({peak_change:+.1f}%)")
        print(f"  Spectral Centroid:  {proc_stats['spectral_centroid']:.1f} Hz ({centroid_change:+.1f}%)")

        print(f"\n[Analysis]")
        if abs(rms_change) > 5:
            if rms_change > 0:
                print(f"  -> Sound volume increased by {rms_change:.0f}%")
            else:
                print(f"  -> Sound volume decreased by {abs(rms_change):.0f}%")
        else:
            print(f"  -> Volume change is minimal")

        if abs(centroid_change) > 5:
            if centroid_change > 0:
                print(f"  -> Spectral centroid shifted up by {centroid_change:.0f}% (brighter/higher)")
            else:
                print(f"  -> Spectral centroid shifted down by {abs(centroid_change):.0f}% (darker/lower)")
        else:
            print(f"  -> Spectral characteristics relatively unchanged")

        print()

    print("=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)


if __name__ == '__main__':
    main()
