"""
Onoma2DSPシステムの使用例
実際にシステムを使う際の基本的な使い方を示します
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.onoma2dsp import Onoma2DSP


def example_1_basic():
    """例1: 基本的な使い方（Attention OFF）"""
    print("=" * 80)
    print("例1: 基本的な使い方")
    print("=" * 80)
    print("\nベルの音を「チリン」から「ゴロゴロ」に変換\n")

    # システムを初期化
    processor = Onoma2DSP(
        model_path='models/rwcp_model.pth',
        scaler_path='models/rwcp_scaler.pkl',
        amplification_factor=1.0,  # DSP効果の強度
        lambda_att=0.0              # Attention機能OFF
    )

    # 変換実行
    result = processor.process(
        source_onomatopoeia='チリン',    # 現在の音のイメージ
        target_onomatopoeia='ゴロゴロ',  # 目標の音のイメージ
        input_audio_path='selected_files/c1/bell2/033.wav',
        output_audio_path='demo_audio/usage_example1.wav',
        verbose=True  # 詳細情報を表示
    )

    print(f"\n[OK] 変換完了: {result['output_audio']}")
    print(f"  特徴量の差分: {result['feature_diff_magnitude']:.2f}")


def example_2_with_attention():
    """例2: Attention機能を使う"""
    print("\n\n" + "=" * 80)
    print("例2: Attention機能を使う")
    print("=" * 80)
    print("\n同じ変換をAttention機能ONで実行\n")

    # Attention機能をONにする
    processor = Onoma2DSP(
        model_path='models/rwcp_model.pth',
        scaler_path='models/rwcp_scaler.pkl',
        amplification_factor=1.0,
        lambda_att=0.7  # Attention機能ON（0.5-0.7を推奨）
    )

    result = processor.process(
        source_onomatopoeia='チリン',
        target_onomatopoeia='ゴロゴロ',
        input_audio_path='selected_files/c1/bell2/033.wav',
        output_audio_path='demo_audio/usage_example2.wav',
        verbose=True
    )

    print(f"\n[OK] 変換完了: {result['output_audio']}")
    print("\n【Attention機能の効果】")
    print("  「チリン」は高音系のオノマトペなので、")
    print("  高音領域（eq_high, eq_presence）の変化が強調されます。")
    print("  結果として、より自然で意図に沿った変換が可能になります。")


def example_3_various_transformations():
    """例3: 様々な変換パターン"""
    print("\n\n" + "=" * 80)
    print("例3: 様々な変換パターン")
    print("=" * 80)

    # 変換パターンのリスト
    transformations = [
        {
            'source': 'コツコツ',
            'target': 'ドンドン',
            'audio': 'selected_files/a1/teak1/001.wav',
            'description': '軽い木の打音 → 重厚な打音'
        },
        {
            'source': 'サラサラ',
            'target': 'ザラザラ',
            'audio': 'selected_files/b1/particl1/003.wav',
            'description': '細かい粒子音 → 粗い粒子音'
        },
        {
            'source': 'カッ',
            'target': 'ガッ',
            'audio': 'selected_files/a1/cherry1/000.wav',
            'description': '清音 → 濁音'
        },
    ]

    processor = Onoma2DSP(
        model_path='models/rwcp_model.pth',
        scaler_path='models/rwcp_scaler.pkl',
        amplification_factor=1.0,
        lambda_att=0.7
    )

    for i, trans in enumerate(transformations, 1):
        print(f"\n[{i}] {trans['description']}")
        print(f"    {trans['source']} → {trans['target']}")

        if not os.path.exists(trans['audio']):
            print(f"    ⚠ ファイルが見つかりません: {trans['audio']}")
            continue

        result = processor.process(
            source_onomatopoeia=trans['source'],
            target_onomatopoeia=trans['target'],
            input_audio_path=trans['audio'],
            output_audio_path=f"demo_audio/usage_example3_{i}.wav",
            verbose=False
        )

        print(f"    [OK] 完了: {result['output_audio']}")


def example_4_parameter_comparison():
    """例4: パラメータ設定の比較"""
    print("\n\n" + "=" * 80)
    print("例4: amplification_factorの比較")
    print("=" * 80)
    print("\n同じ変換を異なる強度で実行して比較\n")

    factors = [3.0, 5.0, 7.0]

    for factor in factors:
        print(f"\n--- amplification_factor = {factor} ---")

        processor = Onoma2DSP(
            model_path='models/rwcp_model.pth',
            scaler_path='models/rwcp_scaler.pkl',
            amplification_factor=factor,
            lambda_att=0.7
        )

        result = processor.process(
            source_onomatopoeia='チリン',
            target_onomatopoeia='ゴロゴロ',
            input_audio_path='selected_files/c1/bell2/033.wav',
            output_audio_path=f'demo_audio/usage_example4_factor{int(factor)}.wav',
            verbose=False
        )

        print(f"[OK] 完了: {result['output_audio']}")

    print("\n【解説】")
    print("  amplification_factorを大きくすると:")
    print("  - DSP効果が強くなります")
    print("  - 変換が劇的になりますが、やりすぎると不自然になる可能性があります")
    print("  - 推奨値: 3.0-7.0（音源や目的に応じて調整）")


def main():
    """メイン関数"""
    print("\n")
    print("=" * 80)
    print(" " * 20 + "Onoma2DSP 使用例デモ")
    print("=" * 80)

    # 出力ディレクトリを作成
    os.makedirs('demo_audio', exist_ok=True)

    # 例1: 基本
    example_1_basic()

    # 例2: Attention
    example_2_with_attention()

    # 例3: 様々な変換
    example_3_various_transformations()

    # 例4: パラメータ比較
    example_4_parameter_comparison()

    print("\n\n" + "=" * 80)
    print("全ての例が完了しました！")
    print("=" * 80)
    print("\n生成されたファイル:")
    print("  demo_audio/usage_example1.wav  (基本)")
    print("  demo_audio/usage_example2.wav  (Attention ON)")
    print("  demo_audio/usage_example3_*.wav (様々な変換)")
    print("  demo_audio/usage_example4_*.wav (強度比較)")
    print("\nこれらのファイルを聴き比べて、効果を確認してください。")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
