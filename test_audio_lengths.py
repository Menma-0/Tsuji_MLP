"""
Test the system with different audio lengths
"""
import sys
import os
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.onoma2dsp import Onoma2DSP

def create_test_audio(duration, sample_rate=44100):
    """Create a simple test audio (sine wave)"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 440Hz (A4 note)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio

def test_duration(duration):
    """Test processing with a specific duration"""
    print(f"\nTesting {duration:.1f} second audio...")

    # Create test audio
    test_file = f'demo_audio/test_{duration:.0f}s.wav'
    audio = create_test_audio(duration)
    sf.write(test_file, audio, 44100)

    # Process
    processor = Onoma2DSP(
        model_path='models/rwcp_model.pth',
        scaler_path='models/rwcp_scaler.pkl',
        amplification_factor=1.0,
        lambda_att=0.7
    )

    output_file = f'demo_audio/test_{duration:.0f}s_processed.wav'

    try:
        import time
        start = time.time()

        result = processor.process(
            source_onomatopoeia='チリン',
            target_onomatopoeia='ゴロゴロ',
            input_audio_path=test_file,
            output_audio_path=output_file,
            verbose=False
        )

        elapsed = time.time() - start

        print(f"  [OK] Processing time: {elapsed:.2f} seconds")
        print(f"  Output: {output_file}")
        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def main():
    print("=" * 80)
    print("Audio Length Compatibility Test")
    print("=" * 80)
    print("\nThis test checks how the system handles different audio lengths.")
    print("Training data: ~0.45 seconds (RWCP-SSD sound effects)")
    print("\nTesting various durations...\n")

    os.makedirs('demo_audio', exist_ok=True)

    # Test different durations
    test_durations = [
        0.5,   # Similar to training data
        1.0,   # 1 second
        2.0,   # 2 seconds
        5.0,   # 5 seconds
        10.0,  # 10 seconds
        30.0,  # 30 seconds (typical music snippet)
    ]

    results = {}

    for duration in test_durations:
        success = test_duration(duration)
        results[duration] = success

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print("\nResults:")
    for duration, success in results.items():
        status = "[OK]" if success else "[FAILED]"
        print(f"  {duration:>5.1f}s: {status}")

    successful = sum(1 for s in results.values() if s)
    print(f"\nSuccessful: {successful}/{len(results)}")

    print("\n" + "=" * 80)
    print("Recommendations:")
    print("=" * 80)
    print("""
このシステムは技術的には任意の長さの音声を処理できますが、
以下の理由から短い音声（効果音）に最適化されています：

1. **学習データ**: RWCP-SSDデータセット（約0.45秒の効果音）
2. **オノマトペの性質**: 短い音のイメージを表現
3. **DSP処理**: 全体に一律のパラメータを適用

【推奨される使用例】
- 効果音: 0.5秒〜3秒（最適）
- 音楽ループ: 2秒〜10秒（可能）
- 長い音声: 10秒以上（技術的に可能だが、全体に同じ処理が適用される）

【長い音声での注意点】
- システムは音声全体に同じDSPパラメータを適用します
- 時間変化する音声（音楽など）では不自然になる可能性があります
- 長い音声の場合、セグメント分割して処理する方が良い結果が得られます

【ベストプラクティス】
- 効果音、打楽器音、短い音素: そのまま使用
- 長い音楽やスピーチ: 意味のある単位で分割して処理
""")
    print("=" * 80)

if __name__ == '__main__':
    main()
