"""
同じ編集操作を複数回適用するテスト
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from src.process_with_pair_model import PairModelProcessor

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    model_path = os.path.join(project_root, 'models', 'pair_model.pth')
    scaler_path = os.path.join(project_root, 'models', 'pair_scaler.pkl')
    output_dir = os.path.join(project_root, 'demo_audio', 'editing_test')

    input_audio = os.path.join(project_root, 'demo_audio', 'diverse_test', '02_coin_original.wav')

    # プロセッサを初期化
    processor = PairModelProcessor(
        model_path=model_path,
        scaler_path=scaler_path
    )

    print("=" * 70)
    print("チャリン → ガチャン を複数回適用するテスト")
    print("=" * 70)

    # 1回目
    print("\n\n" + "=" * 70)
    print("1回目: チャリン → ガチャン")
    print("=" * 70)
    output_1 = os.path.join(output_dir, 'coin_charin_to_gachan_x1.wav')
    result_1 = processor.process(
        source_onomatopoeia='チャリン',
        target_onomatopoeia='ガチャン',
        input_audio_path=input_audio,
        output_audio_path=output_1,
        verbose=True
    )

    # 2回目
    print("\n\n" + "=" * 70)
    print("2回目: チャリン → ガチャン")
    print("=" * 70)
    output_2 = os.path.join(output_dir, 'coin_charin_to_gachan_x2.wav')
    result_2 = processor.process(
        source_onomatopoeia='チャリン',
        target_onomatopoeia='ガチャン',
        input_audio_path=input_audio,
        output_audio_path=output_2,
        verbose=True
    )

    # 3回目
    print("\n\n" + "=" * 70)
    print("3回目: チャリン → ガチャン")
    print("=" * 70)
    output_3 = os.path.join(output_dir, 'coin_charin_to_gachan_x3.wav')
    result_3 = processor.process(
        source_onomatopoeia='チャリン',
        target_onomatopoeia='ガチャン',
        input_audio_path=input_audio,
        output_audio_path=output_3,
        verbose=True
    )

    # サマリー
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results = [
        ('1回目', result_1),
        ('2回目', result_2),
        ('3回目', result_3),
    ]

    print(f"\n{'回数':<8} {'gain':>8} {'eq_sub':>8} {'eq_low':>8} {'eq_mid':>8} {'eq_high':>8} {'attack':>8}")
    print("-" * 70)

    for name, r in results:
        p = r['mapped_params']
        print(f"{name:<8} {p.get('gain_db', 0):>+8.2f} {p.get('eq_sub_db', 0):>+8.2f} "
              f"{p.get('eq_low_db', 0):>+8.2f} {p.get('eq_mid_db', 0):>+8.2f} "
              f"{p.get('eq_high_db', 0):>+8.2f} {p.get('transient_attack', 0):>+8.2f}")


if __name__ == '__main__':
    main()
