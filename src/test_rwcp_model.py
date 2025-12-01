"""
新しく学習したRWCPモデルを使って実際の音声を処理
"""
import pandas as pd
import numpy as np
import os
import sys
import random

# プロジェクトルートを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.onoma2dsp import Onoma2DSP


def select_test_samples(csv_path, n_samples=10):
    """
    テスト用のサンプルを選択

    - 頻出オノマトペから選択
    - 異なる音源タイプを含める
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # ランダムサンプリング（文字化けを避けるため）
    # 異なる音源タイプを選択
    selected = []

    # 各カテゴリから選択
    categories = [
        'a1/cherry', 'a1/magno', 'a1/teak', 'a1/wood',
        'b1/particl', 'b2/particl',
        'c1/bell', 'c1/coin', 'c2/glass', 'c3/key'
    ]

    for category in categories:
        # カテゴリに一致するサンプルを取得
        category_samples = df[df['audio_path'].str.contains(category)]
        if len(category_samples) > 0:
            # 最初の1つを選択
            sample = category_samples.iloc[0]
            audio_path = sample['audio_path'].replace('RWCP-SSD/drysrc/', '../selected_files/')

            if os.path.exists(audio_path):
                selected.append({
                    'onomatopoeia': sample['onomatopoeia'],
                    'audio_path': audio_path,
                    'confidence': sample['confidence'],
                    'acceptability': sample['avg_acceptability']
                })

            if len(selected) >= n_samples:
                break

    return selected


def main():
    print("=" * 80)
    print("TESTING RWCP MODEL WITH REAL AUDIO")
    print("=" * 80)

    # 1. テストサンプルの選択
    print("\n[1] Selecting test samples from CSV...")
    csv_path = '../training_data_jp_utf8bom.csv'
    samples = select_test_samples(csv_path, n_samples=10)

    print(f"Selected {len(samples)} samples:")
    for i, sample in enumerate(samples, 1):
        print(f"  {i}. {sample['onomatopoeia']} - {sample['audio_path']}")

    # 2. 出力フォルダの作成
    output_dir = '../demo_audio/rwcp_test'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[2] Output directory: {output_dir}")

    # 3. モデルのロード
    print("\n[3] Loading RWCP model...")
    processor = Onoma2DSP(
        model_path='../models/rwcp_model.pth',
        scaler_path='../models/rwcp_scaler.pkl'
    )
    print("Model loaded successfully!")

    # 4. 音声処理
    print("\n[4] Processing audio files...")
    print("=" * 80)

    results = []

    for i, sample in enumerate(samples, 1):
        onomatopoeia = sample['onomatopoeia']
        input_path = sample['audio_path']

        # ファイル名を生成
        # selected_files/a1/cherry1/043.wav → a1_cherry1_043
        sound_id = input_path.replace('selected_files/', '').replace('/', '_').replace('.wav', '')

        # 元の音声をコピー
        original_output = os.path.join(output_dir, f"{i:02d}_{sound_id}_original.wav")
        processed_output = os.path.join(output_dir, f"{i:02d}_{sound_id}_{onomatopoeia}.wav")

        print(f"\n[{i}/{len(samples)}] Processing: {onomatopoeia}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {processed_output}")

        try:
            # 元の音声をコピー
            import shutil
            shutil.copy(input_path, original_output)

            # オノマトペで処理
            processor.process(onomatopoeia, input_path, processed_output)

            print(f"  [OK] Processed successfully!")

            results.append({
                'index': i,
                'onomatopoeia': onomatopoeia,
                'sound_id': sound_id,
                'original': original_output,
                'processed': processed_output,
                'confidence': sample['confidence'],
                'acceptability': sample['acceptability']
            })

        except Exception as e:
            print(f"  [ERROR] {e}")

    # 5. 結果のサマリー
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETED")
    print("=" * 80)
    print(f"\nTotal processed: {len(results)} files")
    print(f"Output directory: {output_dir}")

    print("\n[Generated files]:")
    for r in results:
        print(f"  {r['index']}. {r['onomatopoeia']}")
        print(f"     Original:  {os.path.basename(r['original'])}")
        print(f"     Processed: {os.path.basename(r['processed'])}")

    # 6. 聴き比べガイドを作成
    guide_path = os.path.join(output_dir, 'LISTENING_GUIDE.txt')
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("音声処理結果 聞き比べガイド（RWCPモデル）\n")
        f.write("=" * 80 + "\n\n")
        f.write("このフォルダには、RWCP-SSD-Onomatopoeiaデータセットで学習した\n")
        f.write("モデルで処理した音声ファイルが含まれています。\n\n")

        for r in results:
            f.write("=" * 80 + "\n")
            f.write(f"{r['index']}. {r['onomatopoeia']}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"[元の音] {os.path.basename(r['original'])}\n")
            f.write(f"[{r['onomatopoeia']}処理] {os.path.basename(r['processed'])}\n")
            f.write(f"  信頼度: {r['confidence']}/5\n")
            f.write(f"  受容度: {r['acceptability']:.2f}/5\n\n")

        f.write("=" * 80 + "\n")
        f.write("使用モデル\n")
        f.write("=" * 80 + "\n\n")
        f.write("- データセット: RWCP-SSD-Onomatopoeia (8,542サンプル)\n")
        f.write("- モデル: MLP (38 → 32 → 10)\n")
        f.write("- 精度: R² = 0.804, MSE = 0.000900\n")
        f.write("- オノマトペ種類: 1,811種類\n\n")

    print(f"\n[Listening guide created]: {guide_path}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
