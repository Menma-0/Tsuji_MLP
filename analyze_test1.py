"""
test1フォルダの音声処理結果を分析
"""
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os

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

    # ゼロクロスレート（音の粗さの指標）
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    return {
        'duration': duration,
        'rms': rms,
        'peak': peak,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'zcr': zcr
    }

def compare_files():
    """ファイルを比較"""

    comparisons = [
        {
            'name': 'Bell (ベル)',
            'original': 'demo_audio/test1/bell1_original.wav',
            'files': [
                ('キラキラ', 'demo_audio/test1/bell1_kirakira.wav'),
                ('ドンドン', 'demo_audio/test1/bell1_dondon.wav')
            ]
        },
        {
            'name': 'Drum (ドラム)',
            'original': 'demo_audio/test1/drum_original.wav',
            'files': [
                ('ガンガン', 'demo_audio/test1/drum_gangan.wav'),
                ('フワフワ', 'demo_audio/test1/drum_fuwafuwa.wav')
            ]
        },
        {
            'name': 'Coin (コイン)',
            'original': 'demo_audio/test1/coin1_original.wav',
            'files': [
                ('チャリンチャリン', 'demo_audio/test1/coin1_charin.wav')
            ]
        },
        {
            'name': 'Clap (拍手)',
            'original': 'demo_audio/test1/clap1_original.wav',
            'files': [
                ('パチパチ', 'demo_audio/test1/clap1_pachi.wav')
            ]
        },
        {
            'name': 'Sand (砂)',
            'original': 'demo_audio/test1/sand_original.wav',
            'files': [
                ('サラサラ', 'demo_audio/test1/sand_sarasara.wav')
            ]
        },
        {
            'name': 'Whistle (笛)',
            'original': 'demo_audio/test1/whistle_original.wav',
            'files': [
                ('ピーッ', 'demo_audio/test1/whistle_pii.wav')
            ]
        }
    ]

    print("="*80)
    print("AUDIO PROCESSING COMPARISON - TEST1")
    print("="*80)
    print()

    for comp in comparisons:
        print(f"\n{'='*80}")
        print(f"{comp['name']}")
        print(f"{'='*80}")

        # オリジナルを分析
        orig_stats = analyze_audio(comp['original'])
        print(f"\n[Original]")
        print(f"  Duration:           {orig_stats['duration']:.3f}s")
        print(f"  RMS:                {orig_stats['rms']:.4f}")
        print(f"  Peak:               {orig_stats['peak']:.4f}")
        print(f"  Spectral Centroid:  {orig_stats['spectral_centroid']:.1f} Hz")
        print(f"  Spectral Bandwidth: {orig_stats['spectral_bandwidth']:.1f} Hz")
        print(f"  Zero Crossing Rate: {orig_stats['zcr']:.4f}")

        # 処理済みファイルを分析
        for onoma_name, filepath in comp['files']:
            if not os.path.exists(filepath):
                continue

            stats = analyze_audio(filepath)

            # 変化率を計算
            duration_change = ((stats['duration'] - orig_stats['duration']) / orig_stats['duration']) * 100
            rms_change = ((stats['rms'] - orig_stats['rms']) / orig_stats['rms']) * 100
            peak_change = ((stats['peak'] - orig_stats['peak']) / orig_stats['peak']) * 100
            centroid_change = ((stats['spectral_centroid'] - orig_stats['spectral_centroid']) / orig_stats['spectral_centroid']) * 100

            print(f"\n[{onoma_name}]")
            print(f"  Duration:           {stats['duration']:.3f}s ({duration_change:+.1f}%)")
            print(f"  RMS:                {stats['rms']:.4f} ({rms_change:+.1f}%)")
            print(f"  Peak:               {stats['peak']:.4f} ({peak_change:+.1f}%)")
            print(f"  Spectral Centroid:  {stats['spectral_centroid']:.1f} Hz ({centroid_change:+.1f}%)")
            print(f"  Spectral Bandwidth: {stats['spectral_bandwidth']:.1f} Hz")
            print(f"  Zero Crossing Rate: {stats['zcr']:.4f}")

            # 特徴的な変化を解説
            print(f"\n  [Analysis]")
            if rms_change > 50:
                print(f"    → 音量が大幅に増加（+{rms_change:.0f}%）- 力強い音に")
            elif rms_change < -50:
                print(f"    → 音量が大幅に減少（{rms_change:.0f}%）- 柔らかい音に")

            if centroid_change > 10:
                print(f"    → スペクトル重心が上昇（+{centroid_change:.0f}%）- より明るく高音寄りに")
            elif centroid_change < -10:
                print(f"    → スペクトル重心が下降（{centroid_change:.0f}%）- より暗く低音寄りに")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80 + "\n")

if __name__ == '__main__':
    compare_files()
