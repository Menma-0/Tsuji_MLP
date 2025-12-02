"""
Attention機能のテスト

source_onomaから「ユーザがどこを聞いているか」を推定し、
そこを重点的に変化させる機能をテスト
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.onoma2dsp import Onoma2DSP


def main():
    print("=" * 80)
    print("ATTENTION FEATURE TEST")
    print("=" * 80)
    print("\n提案された方法1の実装テスト:")
    print("- source_onomaのDSPテンプレートから注目度ベクトルを作成")
    print("- 注目度が高い次元のΔDSPを強調")
    print()

    # テストケース
    test_cases = [
        {
            'name': 'bell_high_to_low',
            'description': '高音系(チリン)から重低音系(ゴロゴロ)への変換',
            'audio': '../selected_files/c1/bell2/033.wav',
            'source': 'チリン',  # 高音系 → eq_high/presenceが強調されるはず
            'target': 'ゴロゴロ',  # 低音系
            'explanation': 'チリンは高音に注目 → 高音カットが強調されるはず'
        },
        {
            'name': 'bell_low_to_high',
            'description': '重低音系(ゴロゴロ)から高音系(キラキラ)への変換',
            'audio': '../selected_files/c1/bell2/033.wav',
            'source': 'ゴロゴロ',  # 低音系 → eq_sub/lowが強調されるはず
            'target': 'キラキラ',  # 高音系
            'explanation': 'ゴロゴロは低音に注目 → 低音カットが強調されるはず'
        },
        {
            'name': 'wood_sharp_to_dull',
            'description': '鋭い音(カッ)から鈍い音(ガッ)への変換',
            'audio': '../selected_files/a1/wood2/015.wav',
            'source': 'カッ',  # 促音 → transient_attackが強調されるはず
            'target': 'ガッ',
            'explanation': 'カッは促音でアタックに注目 → アタック変化が強調されるはず'
        },
    ]

    for lambda_val in [0.0, 0.5, 1.0]:
        print("\n" + "=" * 80)
        print(f"lambda_att = {lambda_val}")
        print("=" * 80)

        if lambda_val == 0.0:
            print("(Attention機能OFF - 従来の差分ベースのみ)")
        else:
            print(f"(Attention機能ON - 注目度による補正: {lambda_val}倍)")

        # モデルロード
        processor = Onoma2DSP(
            model_path='../models/rwcp_model.pth',
            scaler_path='../models/rwcp_scaler.pkl',
            amplification_factor=5.0,
            lambda_att=lambda_val
        )

        for i, test in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"[{i}] {test['description']}")
            print(f"    {test['explanation']}")
            print(f"{'='*80}")

            output_path = f"../demo_audio/attention_test_{lambda_val:.1f}_{i}.wav"

            result = processor.process(
                test['source'],
                test['target'],
                test['audio'],
                output_path,
                verbose=True
            )

            print(f"\n  [結果サマリー]")
            print(f"    Source: {test['source']}")
            print(f"    Target: {test['target']}")
            print(f"    Feature diff magnitude: {result['feature_diff_magnitude']:.3f}")
            print(f"    Output: {output_path}")

    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)
    print("\n生成されたファイル:")
    print("  lambda_att = 0.0: 従来の差分ベース（ベースライン）")
    print("  lambda_att = 0.5: 中程度のattention補正")
    print("  lambda_att = 1.0: 強いattention補正")
    print("\n各ケースで、source_onomaに応じた重点的な変化が観察できるはずです。")
    print("例: 'チリン'→'ゴロゴロ'では、高音カットがより強調される")


if __name__ == '__main__':
    main()
