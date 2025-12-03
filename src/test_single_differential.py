"""
差分ベースシステムの詳細出力テスト
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.onoma2dsp import Onoma2DSP


def main():
    print("=" * 80)
    print("DIFFERENTIAL SYSTEM - DETAILED TEST")
    print("=" * 80)

    # モデルロード
    processor = Onoma2DSP(
        model_path='../models/rwcp_model.pth',
        scaler_path='../models/rwcp_scaler.pkl',
        amplification_factor=1.0
    )

    # テストケース
    test_cases = [
        {
            'name': 'bell_light_to_heavy',
            'audio': '../selected_files/c1/bell2/033.wav',
            'source': 'チリン',
            'target': 'ゴロゴロ',
            'output': '../demo_audio/test_differential_verbose_1.wav'
        },
        {
            'name': 'bell_clear_to_voiced',
            'audio': '../selected_files/c1/bell2/033.wav',
            'source': 'カラン',
            'target': 'ガンガン',
            'output': '../demo_audio/test_differential_verbose_2.wav'
        },
        {
            'name': 'wood_light_to_heavy',
            'audio': '../selected_files/a1/wood2/015.wav',
            'source': 'カッ',
            'target': 'ガッ',
            'output': '../demo_audio/test_differential_verbose_3.wav'
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n\n{'='*80}")
        print(f"TEST CASE {i}: {test['name']}")
        print(f"{'='*80}")

        result = processor.process(
            test['source'],
            test['target'],
            test['audio'],
            test['output'],
            verbose=True
        )

        print(f"\n[RESULT SUMMARY]")
        print(f"  Feature diff magnitude: {result['feature_diff_magnitude']:.3f}")
        print(f"  Source moras: {result['source_moras']}")
        print(f"  Target moras: {result['target_moras']}")


if __name__ == '__main__':
    main()
