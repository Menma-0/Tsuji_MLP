"""
Attention機能の単一ケース詳細テスト
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.onoma2dsp import Onoma2DSP


def main():
    print("=" * 80)
    print("ATTENTION FEATURE - DETAILED SINGLE CASE TEST")
    print("=" * 80)

    # ケース: チリン → ゴロゴロ
    print("\n【テストケース】")
    print("  音源: ベル")
    print("  source: チリン（高音系 → eq_high/presenceが強調されるはず）")
    print("  target: ゴロゴロ（低音系）")
    print("  期待: チリンが高音に注目しているので、高音カットがより強調される")

    audio = '../selected_files/c1/bell2/033.wav'

    # lambda_att = 0.0（ベースライン）
    print("\n" + "=" * 80)
    print("【1】lambda_att = 0.0（Attention OFF）")
    print("=" * 80)

    processor_baseline = Onoma2DSP(
        model_path='../models/rwcp_model.pth',
        scaler_path='../models/rwcp_scaler.pkl',
        amplification_factor=5.0,
        lambda_att=0.0
    )

    result_baseline = processor_baseline.process(
        'チリン', 'ゴロゴロ',
        audio,
        '../demo_audio/attention_single_baseline.wav',
        verbose=True
    )

    # lambda_att = 0.7（Attention ON）
    print("\n" + "=" * 80)
    print("【2】lambda_att = 0.7（Attention ON）")
    print("=" * 80)

    processor_attention = Onoma2DSP(
        model_path='../models/rwcp_model.pth',
        scaler_path='../models/rwcp_scaler.pkl',
        amplification_factor=5.0,
        lambda_att=0.7
    )

    result_attention = processor_attention.process(
        'チリン', 'ゴロゴロ',
        audio,
        '../demo_audio/attention_single_with_attention.wav',
        verbose=True
    )

    # 比較
    print("\n" + "=" * 80)
    print("【3】パラメータ比較")
    print("=" * 80)

    params_baseline = result_baseline['mapped_params']
    params_attention = result_attention['mapped_params']

    print(f"\n{'Parameter':<25} {'Baseline':>12} {'Attention':>12} {'Change':>12}")
    print("-" * 65)

    for key in params_baseline.keys():
        val_base = params_baseline[key]
        val_att = params_attention[key]
        change = val_att - val_base

        # パーセント表示
        if abs(val_base) > 0.01:
            pct_change = (change / val_base) * 100
            print(f"{key:<25} {val_base:>12.2f} {val_att:>12.2f} {change:>+11.2f} ({pct_change:+.1f}%)")
        else:
            print(f"{key:<25} {val_base:>12.2f} {val_att:>12.2f} {change:>+11.2f}")

    print("\n" + "=" * 80)
    print("【4】解釈")
    print("=" * 80)
    print("\nチリンは高音系オノマトペなので、")
    print("- eq_high / eq_presence の注目度が高い")
    print("- チリン→ゴロゴロ変換では高音をカットする必要がある")
    print("- Attention機能により、高音カットがより強調される")
    print("\n結果:")
    print("- eq_high_db / eq_presence_db の変化がより大きくなっているはず")
    print("- 低音系(eq_sub/low)は元々チリンで注目していないので変化は控えめ")


if __name__ == '__main__':
    main()
