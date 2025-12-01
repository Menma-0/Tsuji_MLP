"""
多様なオノマトペで音声処理テスト
"""
import pandas as pd
import numpy as np
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.onoma2dsp import Onoma2DSP


def select_diverse_samples(csv_path):
    """
    多様なオノマトペを手動で選択

    - 濁音（ガ、ザ、ダ、バ）
    - 促音（ッ）
    - 長音（ー）
    - 繰り返し
    など、様々なパターンを含む
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 手動で多様なオノマトペを指定
    # CSVに実際に存在するもの
    target_patterns = [
        # 同じ音源で異なるオノマトペを試す
        ('selected_files/c1/bell1', ['original']),  # ベル
        ('selected_files/c1/coin1', ['original']),  # コイン
        ('selected_files/a1/wood1', ['original']),  # 木材
        ('selected_files/b1/particl1', ['original']),  # 粒子
        ('selected_files/c3/whistle1', ['original']),  # 笛
        ('selected_files/c4/buzzer', ['original']),  # ブザー
        ('selected_files/b5/clap1', ['original']),  # 拍手
    ]

    # 各音源タイプから1つずつ取得
    selected = []

    # 音源カテゴリ別に選択
    categories = {
        'bell': 'c1/bell',
        'coin': 'c1/coin',
        'wood': 'a1/wood',
        'particle': 'b1/particl',
        'whistle': 'c3/whistle',
        'buzzer': 'c4/buzzer',
        'clap': 'b5/clap',
        'drum': 'a4/drum',
        'glass': 'a4/glass',
        'metal': 'a2/bowl',
    }

    for name, category in categories.items():
        samples = df[df['audio_path'].str.contains(category)]
        if len(samples) > 0:
            # 最初のサンプル
            sample = samples.iloc[0]
            audio_path = sample['audio_path'].replace('RWCP-SSD/drysrc/', '../selected_files/')

            if os.path.exists(audio_path):
                selected.append({
                    'name': name,
                    'onomatopoeia': sample['onomatopoeia'],
                    'audio_path': audio_path,
                    'confidence': sample['confidence'],
                    'acceptability': sample['avg_acceptability']
                })

    return selected


def main():
    print("=" * 80)
    print("TESTING WITH DIVERSE ONOMATOPOEIA")
    print("=" * 80)

    # 1. サンプル選択
    print("\n[1] Selecting diverse audio samples...")
    csv_path = '../training_data_jp_utf8bom.csv'
    samples = select_diverse_samples(csv_path)

    print(f"Selected {len(samples)} samples\n")

    # 2. 出力フォルダ（増幅版）
    output_dir = '../demo_audio/diverse_test_amplified'
    os.makedirs(output_dir, exist_ok=True)

    # 3. モデルロード
    print("\n[2] Loading RWCP model...")
    processor = Onoma2DSP(
        model_path='../models/rwcp_model.pth',
        scaler_path='../models/rwcp_scaler.pkl'
    )

    # 4. 各音源に対して複数のオノマトペで処理
    print("\n[3] Processing with multiple onomatopoeia...")
    print("=" * 80)

    # 処理するオノマトペのリスト（多様性重視）
    test_onomatopoeia = [
        'カッ',         # 促音のみ（清音）
        'ガッ',         # 促音 + 濁音
        'ゴロゴロ',     # 濁音 + 繰り返し
        'サラサラ',     # 繰り返し（清音）
        'キラキラ',     # 繰り返し（清音・高音）
        'ドンドン',     # 濁音 + 繰り返し
        'ピーッ',       # 長音 + 促音
        'ガンガン',     # 濁音 + 繰り返し
        'チリンチリン', # 繰り返し
        'バシャバシャ', # 濁音 + 繰り返し
    ]

    results = []

    for i, sample in enumerate(samples[:3], 1):  # 最初の3つの音源で試す
        name = sample['name']
        input_path = sample['audio_path']

        print(f"\n{'='*80}")
        print(f"[{i}] Sound: {name}")
        print(f"    Input: {input_path}")
        print(f"{'='*80}\n")

        # 元の音声をコピー
        original_output = os.path.join(output_dir, f"{i:02d}_{name}_original.wav")
        shutil.copy(input_path, original_output)

        # 複数のオノマトペで処理
        for j, onoma in enumerate(test_onomatopoeia, 1):
            processed_output = os.path.join(output_dir, f"{i:02d}_{name}_{j:02d}_{onoma}.wav")

            print(f"  [{j}/10] {onoma}...", end=' ')

            try:
                processor.process(onoma, input_path, processed_output)
                print("[OK]")

                results.append({
                    'sound': name,
                    'onomatopoeia': onoma,
                    'original': original_output,
                    'processed': processed_output
                })

            except Exception as e:
                print(f"[ERROR: {e}]")

    # 5. 聴き比べガイド作成
    guide_path = os.path.join(output_dir, 'LISTENING_GUIDE.txt')
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("多様なオノマトペによる音声処理結果\n")
        f.write("=" * 80 + "\n\n")
        f.write("同じ音源に対して、異なるオノマトペで処理した結果を比較できます。\n\n")

        # 音源ごとにグループ化
        sounds = list(set([r['sound'] for r in results]))

        for sound in sounds:
            f.write("=" * 80 + "\n")
            f.write(f"{sound.upper()}\n")
            f.write("=" * 80 + "\n\n")

            sound_results = [r for r in results if r['sound'] == sound]
            f.write(f"[元の音] {os.path.basename(sound_results[0]['original'])}\n\n")

            for r in sound_results:
                f.write(f"[{r['onomatopoeia']}] {os.path.basename(r['processed'])}\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("推奨の聴き比べ方法\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. 元の音を聴く\n")
        f.write("2. 各オノマトペで処理した音を順番に聴く\n")
        f.write("3. 特に以下の違いに注目：\n")
        f.write("   - カッ vs ガッ: 清音 vs 濁音の違い\n")
        f.write("   - サラサラ vs ゴロゴロ: 軽やかさ vs 重厚さ\n")
        f.write("   - キラキラ vs ガンガン: 高音強調 vs 低音強調\n")
        f.write("   - ピーッ: 長音による引き伸ばし効果\n\n")

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETED")
    print("=" * 80)
    print(f"\nTotal processed: {len(results)} files")
    print(f"Output directory: {output_dir}")
    print(f"Listening guide: {guide_path}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
