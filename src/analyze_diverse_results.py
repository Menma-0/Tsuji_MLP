"""
多様なオノマトペによる音声処理結果を詳細に分析
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
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

    # Zero crossing rate (音の粗さの指標)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    return {
        'duration': duration,
        'rms': rms,
        'peak': peak,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'zcr': zcr
    }


def main():
    print("=" * 80)
    print("DIVERSE ONOMATOPOEIA PROCESSING ANALYSIS")
    print("=" * 80)

    output_dir = '../demo_audio/diverse_test'

    # オノマトペのリスト（順番を保持）
    test_onomatopoeia = [
        'カッ',         # 促音のみ（清音）
        'ガッ',         # 促音 + 濁音
        'ゴロゴロ',     # 濁音 + 繰り返し
        'サラサラ',     # 繰り返し（清音）
        'キラキラ',     # 繰り返し（清音・高音）
        'ドンドン',     # 濁音 + 繰り返し
        'ピーッ',       # 長音 + 促音
        'ガンガン',     # 濁音 + 繰り返し
        'チリンチリン', # 繰り返し
        'バシャバシャ', # 濁音 + 繰り返し
    ]

    # 音源ごとに分析
    original_files = sorted(glob.glob(os.path.join(output_dir, '*_original.wav')))

    print(f"\nFound {len(original_files)} audio sources\n")

    for orig_file in original_files:
        # 音源名を取得
        base_name = os.path.basename(orig_file).replace('_original.wav', '')
        sound_name = base_name.split('_')[1]  # 01_bell → bell

        print("=" * 80)
        print(f"SOUND: {sound_name.upper()}")
        print("=" * 80)
        print(f"Base file: {os.path.basename(orig_file)}\n")

        # 元の音声を分析
        orig_stats = analyze_audio(orig_file)

        print("[ORIGINAL]")
        print(f"  RMS:                {orig_stats['rms']:.4f}")
        print(f"  Peak:               {orig_stats['peak']:.4f}")
        print(f"  Spectral Centroid:  {orig_stats['spectral_centroid']:.1f} Hz")
        print(f"  Spectral Bandwidth: {orig_stats['spectral_bandwidth']:.1f} Hz")
        print(f"  Zero Crossing Rate: {orig_stats['zcr']:.4f}")
        print()

        # 各オノマトペで処理した結果を分析
        print(f"{'Onomatopoeia':<15} {'RMS%':>8} {'Peak%':>8} {'Centroid%':>12} {'Bandwidth%':>12} {'ZCR%':>8}")
        print("-" * 80)

        results = []

        for i, onoma in enumerate(test_onomatopoeia, 1):
            # 処理済みファイルを探す
            processed_pattern = os.path.join(output_dir, f"{base_name}_{i:02d}_{onoma}.wav")

            if not os.path.exists(processed_pattern):
                continue

            # 分析
            proc_stats = analyze_audio(processed_pattern)

            # 変化率を計算
            rms_change = ((proc_stats['rms'] - orig_stats['rms']) / orig_stats['rms']) * 100
            peak_change = ((proc_stats['peak'] - orig_stats['peak']) / orig_stats['peak']) * 100
            centroid_change = ((proc_stats['spectral_centroid'] - orig_stats['spectral_centroid']) / orig_stats['spectral_centroid']) * 100
            bandwidth_change = ((proc_stats['spectral_bandwidth'] - orig_stats['spectral_bandwidth']) / orig_stats['spectral_bandwidth']) * 100
            zcr_change = ((proc_stats['zcr'] - orig_stats['zcr']) / orig_stats['zcr']) * 100

            print(f"{onoma:<15} {rms_change:>+7.1f}% {peak_change:>+7.1f}% {centroid_change:>+11.1f}% {bandwidth_change:>+11.1f}% {zcr_change:>+7.1f}%")

            results.append({
                'onoma': onoma,
                'rms_change': rms_change,
                'centroid_change': centroid_change,
                'bandwidth_change': bandwidth_change,
                'zcr_change': zcr_change
            })

        # 主要な傾向を分析
        print("\n[KEY OBSERVATIONS]")

        # 濁音 vs 清音
        ka_result = next((r for r in results if r['onoma'] == 'カッ'), None)
        ga_result = next((r for r in results if r['onoma'] == 'ガッ'), None)

        if ka_result and ga_result:
            print(f"\n  [清音 vs 濁音] カッ vs ガッ")
            print(f"    カッ: RMS {ka_result['rms_change']:+.1f}%, Centroid {ka_result['centroid_change']:+.1f}%")
            print(f"    ガッ: RMS {ga_result['rms_change']:+.1f}%, Centroid {ga_result['centroid_change']:+.1f}%")
            if ga_result['rms_change'] > ka_result['rms_change']:
                print(f"    -> 濁音(ガッ)は音量が大きくなる傾向")
            if ga_result['centroid_change'] < ka_result['centroid_change']:
                print(f"    -> 濁音(ガッ)は低音が強調される傾向")

        # 軽やか vs 重厚
        sara_result = next((r for r in results if r['onoma'] == 'サラサラ'), None)
        goro_result = next((r for r in results if r['onoma'] == 'ゴロゴロ'), None)

        if sara_result and goro_result:
            print(f"\n  [軽やか vs 重厚] サラサラ vs ゴロゴロ")
            print(f"    サラサラ: Centroid {sara_result['centroid_change']:+.1f}%, Bandwidth {sara_result['bandwidth_change']:+.1f}%")
            print(f"    ゴロゴロ: Centroid {goro_result['centroid_change']:+.1f}%, Bandwidth {goro_result['bandwidth_change']:+.1f}%")
            if sara_result['centroid_change'] > goro_result['centroid_change']:
                print(f"    -> サラサラは高音域が強調される")
            if goro_result['bandwidth_change'] > sara_result['bandwidth_change']:
                print(f"    -> ゴロゴロは周波数帯域が広がる")

        # 高音強調 vs 低音強調
        kira_result = next((r for r in results if r['onoma'] == 'キラキラ'), None)
        gan_result = next((r for r in results if r['onoma'] == 'ガンガン'), None)

        if kira_result and gan_result:
            print(f"\n  [高音 vs 低音] キラキラ vs ガンガン")
            print(f"    キラキラ: Centroid {kira_result['centroid_change']:+.1f}%")
            print(f"    ガンガン: Centroid {gan_result['centroid_change']:+.1f}%")
            if kira_result['centroid_change'] > gan_result['centroid_change']:
                print(f"    -> キラキラは明るく高音を強調")
            if gan_result['centroid_change'] < kira_result['centroid_change']:
                print(f"    -> ガンガンは低音を強調")

        # 長音効果
        pii_result = next((r for r in results if r['onoma'] == 'ピーッ'), None)

        if pii_result:
            print(f"\n  [長音効果] ピーッ")
            print(f"    Centroid {pii_result['centroid_change']:+.1f}%, Bandwidth {pii_result['bandwidth_change']:+.1f}%")
            if pii_result['centroid_change'] > 0:
                print(f"    -> 長音により高音域が引き伸ばされる効果")

        print("\n")

    print("=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)


if __name__ == '__main__':
    main()
