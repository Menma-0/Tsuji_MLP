"""
デモ音声ファイルを作成するスクリプト
"""
import numpy as np
import soundfile as sf
import os

# デモ音声を作成
os.makedirs('demo_audio', exist_ok=True)

# サイン波ベースの音を生成（より豊かな音）
sample_rate = 44100
duration = 3.0
t = np.linspace(0, duration, int(sample_rate * duration))

# 複数の倍音を含む音
audio = (
    0.4 * np.sin(2 * np.pi * 220 * t) +      # A3
    0.25 * np.sin(2 * np.pi * 440 * t) +     # A4
    0.15 * np.sin(2 * np.pi * 660 * t) +     # E5
    0.1 * np.sin(2 * np.pi * 880 * t) +      # A5
    0.05 * np.sin(2 * np.pi * 1320 * t)      # E6
)

# ノイズを少し追加（質感を出すため）
np.random.seed(42)
noise = np.random.randn(len(t)) * 0.05
audio = audio + noise

# エンベロープを適用
attack = 0.05
decay = 0.3
sustain_level = 0.7
release = 1.0

envelope = np.ones_like(t)
attack_samples = int(attack * sample_rate)
decay_samples = int(decay * sample_rate)
release_samples = int(release * sample_rate)

# アタック
envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
# ディケイ
envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain_level, decay_samples)
# リリース
envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)

audio = audio * envelope

# ノーマライズ
audio = audio / np.max(np.abs(audio)) * 0.7

# 保存
sf.write('demo_audio/original.wav', audio, sample_rate)
print(f'Demo audio created: demo_audio/original.wav')
print(f'Duration: {duration}s, Sample rate: {sample_rate}Hz')
print(f'Peak amplitude: {np.max(np.abs(audio)):.3f}')
