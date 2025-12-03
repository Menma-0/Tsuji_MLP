"""
Attention機能の包括的テスト
多様な音源とオノマトペの組み合わせで検証
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.onoma2dsp import Onoma2DSP


def main():
    print("=" * 80)
    print("COMPREHENSIVE ATTENTION FEATURE TEST")
    print("=" * 80)
    print("\n多様な音源とオノマトペでAttention機能をテスト")
    print("- 異なる音源カテゴリ")
    print("- 異なるオノマトペパターン")
    print("- lambda_att = 0.0（OFF）vs 0.7（ON）の比較\n")

    # テストケース
    test_cases = [
        {
            'category': 'Metal (金属)',
            'audio': '../selected_files/a2/bowl1/030.wav',
            'cases': [
                ('カーン', 'ゴンゴン', '高音金属音 → 重厚な低音'),
                ('キーン', 'ドーン', '鋭い高音 → 鈍い重低音'),
                ('チン', 'ガンガン', '軽い音 → 激しい音'),
            ]
        },
        {
            'category': 'Glass (ガラス)',
            'audio': '../selected_files/a4/glass1/051.wav',
            'cases': [
                ('カチャン', 'ドスン', 'ガラス音 → 重い音'),
                ('チリン', 'ゴロゴロ', '軽やか → 重厚'),
                ('キラキラ', 'ザラザラ', '明るい → 粗い'),
            ]
        },
        {
            'category': 'Wood (木材)',
            'audio': '../selected_files/a1/teak1/001.wav',
            'cases': [
                ('コツコツ', 'ドンドン', '軽い打音 → 重い打音'),
                ('カッ', 'ゴッ', '鋭い → 鈍い'),
                ('トントン', 'ガンガン', 'リズミカル → 激しい'),
            ]
        },
        {
            'category': 'Particle (粒子)',
            'audio': '../selected_files/b1/particl1/003.wav',
            'cases': [
                ('サラサラ', 'ザラザラ', '細かい → 粗い'),
                ('シャラシャラ', 'ガラガラ', '軽やか → 重い'),
                ('カラカラ', 'ゴロゴロ', '乾いた → 重厚'),
            ]
        },
        {
            'category': 'Drum (太鼓)',
            'audio': '../selected_files/a4/drum1/002.wav',
            'cases': [
                ('トン', 'ドーン', '短い → 長く重い'),
                ('タン', 'ダン', '軽い → 重い'),
                ('ポン', 'ドン', '柔らかい → 硬い'),
            ]
        },
    ]

    # 出力ディレクトリ
    output_dir = '../demo_audio/attention_comprehensive'
    os.makedirs(output_dir, exist_ok=True)

    # 結果を保存
    results = []

    for cat_idx, category_data in enumerate(test_cases, 1):
        category = category_data['category']
        audio_path = category_data['audio']

        # 音声ファイルの存在確認
        if not os.path.exists(audio_path):
            print(f"\n[SKIP] {category}: Audio file not found")
            continue

        print(f"\n{'='*80}")
        print(f"[{cat_idx}] {category}")
        print(f"    Audio: {audio_path}")
        print(f"{'='*80}")

        for case_idx, (source, target, description) in enumerate(category_data['cases'], 1):
            print(f"\n  [{case_idx}] {description}")
            print(f"      {source} → {target}")

            # lambda_att = 0.0 (Baseline)
            processor_baseline = Onoma2DSP(
                model_path='../models/rwcp_model.pth',
                scaler_path='../models/rwcp_scaler.pkl',
                amplification_factor=1.0,
                lambda_att=0.0
            )

            output_baseline = os.path.join(
                output_dir,
                f"{cat_idx:02d}_{case_idx:02d}_baseline_{source}_to_{target}.wav"
            )

            try:
                result_baseline = processor_baseline.process(
                    source, target, audio_path, output_baseline, verbose=False
                )
                print(f"      [Baseline] OK - magnitude: {result_baseline['feature_diff_magnitude']:.2f}")
            except Exception as e:
                print(f"      [Baseline] ERROR: {e}")
                continue

            # lambda_att = 0.7 (With Attention)
            processor_attention = Onoma2DSP(
                model_path='../models/rwcp_model.pth',
                scaler_path='../models/rwcp_scaler.pkl',
                amplification_factor=1.0,
                lambda_att=0.7
            )

            output_attention = os.path.join(
                output_dir,
                f"{cat_idx:02d}_{case_idx:02d}_attention_{source}_to_{target}.wav"
            )

            try:
                result_attention = processor_attention.process(
                    source, target, audio_path, output_attention, verbose=False
                )
                print(f"      [Attention] OK")

                # パラメータ比較
                params_base = result_baseline['mapped_params']
                params_att = result_attention['mapped_params']

                # 最大変化を記録
                max_change_key = None
                max_change_val = 0
                for key in params_base.keys():
                    if 'db' in key:
                        change = abs(params_att[key] - params_base[key])
                        if change > max_change_val:
                            max_change_val = change
                            max_change_key = key

                print(f"      [Max Change] {max_change_key}: {max_change_val:.2f} dB")

                # 結果を記録
                results.append({
                    'category': category,
                    'source': source,
                    'target': target,
                    'description': description,
                    'max_change_param': max_change_key,
                    'max_change_value': max_change_val,
                    'output_baseline': output_baseline,
                    'output_attention': output_attention
                })

            except Exception as e:
                print(f"      [Attention] ERROR: {e}")
                continue

    # サマリーレポート
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\nTotal test cases: {len(results)}")
    print(f"Output directory: {output_dir}\n")

    # 結果をDataFrameに変換
    if results:
        df = pd.DataFrame(results)

        print("Most affected parameters by Attention:")
        param_counts = df['max_change_param'].value_counts()
        for param, count in param_counts.items():
            print(f"  {param:<25}: {count} cases")

        print(f"\nAverage max change: {df['max_change_value'].mean():.2f} dB")
        print(f"Max change observed: {df['max_change_value'].max():.2f} dB")

        # カテゴリ別サマリー
        print("\nBy category:")
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            print(f"  {category:<20}: {len(cat_df)} cases, avg change: {cat_df['max_change_value'].mean():.2f} dB")

    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST COMPLETED")
    print("=" * 80)


if __name__ == '__main__':
    main()
