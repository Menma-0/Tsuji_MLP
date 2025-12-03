"""
Check audio file durations in the dataset
"""
import librosa
import glob
import numpy as np

# Get all audio files
files = glob.glob('selected_files/**/*.wav', recursive=True)

print(f"Total files found: {len(files)}")

# Sample 50 files
sample_files = files[:50] if len(files) > 50 else files

durations = []
for f in sample_files:
    try:
        dur = librosa.get_duration(path=f)
        durations.append(dur)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if durations:
    print(f"\nSample size: {len(durations)} files")
    print(f"Average duration: {np.mean(durations):.2f} seconds")
    print(f"Shortest: {np.min(durations):.2f} seconds")
    print(f"Longest: {np.max(durations):.2f} seconds")
    print(f"Median: {np.median(durations):.2f} seconds")
    print(f"Std dev: {np.std(durations):.2f} seconds")

    print("\nDuration distribution:")
    bins = [0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float('inf')]
    labels = ['<0.5s', '0.5-1s', '1-2s', '2-3s', '3-5s', '5-10s', '>10s']

    for i in range(len(bins)-1):
        count = sum(1 for d in durations if bins[i] <= d < bins[i+1])
        pct = (count / len(durations)) * 100
        print(f"  {labels[i]:>8}: {count:3d} files ({pct:5.1f}%)")

    print("\nFirst 10 files:")
    for i, (f, d) in enumerate(zip(sample_files[:10], durations[:10]), 1):
        filename = f.replace('selected_files/', '')
        print(f"  {i:2d}. {filename:<40} {d:.2f}s")
