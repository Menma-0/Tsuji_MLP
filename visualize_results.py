"""
音声処理結果を可視化するスクリプト
"""
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os

def analyze_audio(filepath):
    """音声ファイルを分析"""
    audio, sr = librosa.load(filepath, sr=44100)

    # 基本統計
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    duration = len(audio) / sr

    # スペクトル重心
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    return {
        'duration': duration,
        'rms': rms,
        'peak': peak,
        'spectral_centroid': spectral_centroid
    }

def plot_comparison():
    """処理結果を比較プロット"""
    # ファイルリスト
    files = {
        'Original': 'demo_audio/original.wav',
        'ガンガン': 'demo_audio/output_gangan.wav',
        'サラサラ': 'demo_audio/output_sarasara.wav',
        'キラキラ': 'demo_audio/output_kirakira.wav',
        'ドンドン': 'demo_audio/output_dondon.wav',
        'フワフワ': 'demo_audio/output_fuwafuwa.wav'
    }

    # 分析
    results = {}
    for name, filepath in files.items():
        if os.path.exists(filepath):
            results[name] = analyze_audio(filepath)
            print(f"{name}:")
            print(f"  Duration: {results[name]['duration']:.2f}s")
            print(f"  RMS: {results[name]['rms']:.4f}")
            print(f"  Peak: {results[name]['peak']:.4f}")
            print(f"  Spectral Centroid: {results[name]['spectral_centroid']:.1f} Hz")
            print()

    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = list(results.keys())
    colors = ['gray', 'red', 'blue', 'gold', 'purple', 'green']

    # 1. 持続時間
    ax = axes[0, 0]
    durations = [results[name]['duration'] for name in names]
    ax.bar(range(len(names)), durations, color=colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Duration (s)')
    ax.set_title('Audio Duration Comparison')
    ax.grid(True, alpha=0.3)

    # 2. RMS（音量）
    ax = axes[0, 1]
    rms_values = [results[name]['rms'] for name in names]
    ax.bar(range(len(names)), rms_values, color=colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('RMS')
    ax.set_title('RMS Level Comparison')
    ax.grid(True, alpha=0.3)

    # 3. ピーク
    ax = axes[1, 0]
    peak_values = [results[name]['peak'] for name in names]
    ax.bar(range(len(names)), peak_values, color=colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Peak Amplitude')
    ax.set_title('Peak Amplitude Comparison')
    ax.grid(True, alpha=0.3)

    # 4. スペクトル重心（明るさ）
    ax = axes[1, 1]
    centroids = [results[name]['spectral_centroid'] for name in names]
    ax.bar(range(len(names)), centroids, color=colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Spectral Centroid (Hz)')
    ax.set_title('Spectral Centroid Comparison (Brightness)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_audio/comparison_plot.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to: demo_audio/comparison_plot.png")

    # 波形プロット
    fig, axes = plt.subplots(len(files), 1, figsize=(12, 10), sharex=True)

    for idx, (name, filepath) in enumerate(files.items()):
        if os.path.exists(filepath):
            audio, sr = librosa.load(filepath, sr=44100)
            t = np.linspace(0, len(audio)/sr, len(audio))

            axes[idx].plot(t, audio, linewidth=0.5, color=colors[idx])
            axes[idx].set_ylabel(name)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim(-1, 1)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Waveform Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('demo_audio/waveform_comparison.png', dpi=150, bbox_inches='tight')
    print("Waveform plot saved to: demo_audio/waveform_comparison.png")

if __name__ == '__main__':
    print("="*60)
    print("AUDIO PROCESSING RESULTS ANALYSIS")
    print("="*60 + "\n")

    plot_comparison()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETED!")
    print("="*60)
