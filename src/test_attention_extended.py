"""
Attention機能の拡張テスト
selected_filesから実際に存在するファイルを使用
"""
import sys
import os
import glob
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.onoma2dsp import Onoma2DSP


def find_audio_files():
    """selected_filesから音声ファイルを検索"""
    base_dir = '../selected_files'
    audio_files = []

    # 各カテゴリから1つずつ取得
    categories = ['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b5', 'c1', 'c2', 'c3', 'c4']

    for cat in categories:
        pattern = os.path.join(base_dir, cat, '**', '*.wav')
        files = glob.glob(pattern, recursive=True)
        if files:
            # 各カテゴリから最初の2ファイル
            audio_files.extend(files[:2])

    return audio_files


def main():
    print("=" * 80)
    print("EXTENDED ATTENTION FEATURE TEST")
    print("=" * 80)
    print("\nselected_filesから実際のファイルを検索してテスト\n")

    # 音声ファイルを検索
    audio_files = find_audio_files()
    print(f"Found {len(audio_files)} audio files\n")

    if len(audio_files) == 0:
        print("ERROR: No audio files found")
        return

    # テストするオノマトペの組み合わせ
    onoma_patterns = [
        ('チリン', 'ゴロゴロ', '高音 → 低音'),
        ('カッ', 'ガッ', '清音 → 濁音'),
        ('コツコツ', 'ドンドン', '軽快 → 重厚'),
        ('サラサラ', 'ザラザラ', '細かい → 粗い'),
        ('キラキラ', 'ガンガン', '明るい → 激しい'),
    ]

    # 出力ディレクトリ
    output_dir = '../demo_audio/attention_extended'
    os.makedirs(output_dir, exist_ok=True)

    # 結果を保存
    results = []

    # 各音声ファイルで最初のオノマトペパターンのみテスト
    for file_idx, audio_path in enumerate(audio_files[:10], 1):  # 最初の10ファイル
        # ファイル名から音源情報を取得
        rel_path = audio_path.replace('../selected_files/', '')
        category = rel_path.split('/')[0]

        print(f"[{file_idx}/{min(10, len(audio_files))}] {rel_path}")

        # 最初のオノマトペパターンのみ
        source, target, description = onoma_patterns[0]
        print(f"    {source} → {target} ({description})")

        # Baseline
        processor_baseline = Onoma2DSP(
            model_path='../models/rwcp_model.pth',
            scaler_path='../models/rwcp_scaler.pkl',
            amplification_factor=5.0,
            lambda_att=0.0
        )

        output_baseline = os.path.join(
            output_dir,
            f"{file_idx:02d}_baseline_{category}.wav"
        )

        try:
            result_baseline = processor_baseline.process(
                source, target, audio_path, output_baseline, verbose=False
            )
            print(f"    [Baseline] OK")
        except Exception as e:
            print(f"    [Baseline] ERROR: {e}")
            continue

        # With Attention
        processor_attention = Onoma2DSP(
            model_path='../models/rwcp_model.pth',
            scaler_path='../models/rwcp_scaler.pkl',
            amplification_factor=5.0,
            lambda_att=0.7
        )

        output_attention = os.path.join(
            output_dir,
            f"{file_idx:02d}_attention_{category}.wav"
        )

        try:
            result_attention = processor_attention.process(
                source, target, audio_path, output_attention, verbose=False
            )
            print(f"    [Attention] OK")

            # パラメータ比較
            params_base = result_baseline['mapped_params']
            params_att = result_attention['mapped_params']

            # 変化を計算
            changes = {}
            for key in params_base.keys():
                if 'db' in key:
                    changes[key] = abs(params_att[key] - params_base[key])

            # 最大変化
            max_key = max(changes, key=changes.get)
            max_val = changes[max_key]

            print(f"    [Max Change] {max_key}: {max_val:.2f} dB")

            results.append({
                'file': rel_path,
                'category': category,
                'source': source,
                'target': target,
                'max_change_param': max_key,
                'max_change_value': max_val,
                'feature_diff_mag': result_baseline['feature_diff_magnitude']
            })

        except Exception as e:
            print(f"    [Attention] ERROR: {e}")
            continue

    # サマリーレポート
    print("\n" + "=" * 80)
    print("EXTENDED TEST SUMMARY")
    print("=" * 80)
    print(f"\nTotal successful cases: {len(results)}")
    print(f"Output directory: {output_dir}\n")

    if results:
        df = pd.DataFrame(results)

        print("Statistics:")
        print(f"  Average max change: {df['max_change_value'].mean():.2f} dB")
        print(f"  Max change observed: {df['max_change_value'].max():.2f} dB")
        print(f"  Min change observed: {df['max_change_value'].min():.2f} dB")
        print(f"  Std deviation: {df['max_change_value'].std():.2f} dB")

        print("\nMost affected parameters:")
        param_counts = df['max_change_param'].value_counts()
        for param, count in param_counts.head(5).items():
            avg_change = df[df['max_change_param'] == param]['max_change_value'].mean()
            print(f"  {param:<25}: {count} cases (avg: {avg_change:.2f} dB)")

        print("\nBy category:")
        for category in sorted(df['category'].unique()):
            cat_df = df[df['category'] == category]
            print(f"  {category:<10}: {len(cat_df)} files, avg change: {cat_df['max_change_value'].mean():.2f} dB")

        # 変化が大きかったケース Top 5
        print("\nTop 5 largest changes:")
        top5 = df.nlargest(5, 'max_change_value')
        for idx, row in top5.iterrows():
            print(f"  {row['file']:<40} {row['max_change_param']:<20} {row['max_change_value']:.2f} dB")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
