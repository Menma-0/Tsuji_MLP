"""
差分ベースのオノマトペシステムをテスト

現在の音を表すオノマトペ → 目標の音を表すオノマトペへの変換
"""
import pandas as pd
import numpy as np
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.onoma2dsp import Onoma2DSP


def main():
    print("=" * 80)
    print("DIFFERENTIAL ONOMATOPOEIA SYSTEM TEST")
    print("=" * 80)

    # 1. テスト音源を選択
    print("\n[1] Selecting test audio samples...")

    test_cases = [
        {
            'name': 'bell',
            'audio_path': '../selected_files/c1/bell2/033.wav',
            'transformations': [
                ('チリン', 'ゴロゴロ'),  # 軽やかな音 → 重厚な音
                ('チリン', 'ガンガン'),  # 軽やかな音 → 激しい音
                ('チリン', 'サラサラ'),  # 軽やかな音 → より軽やかに
                ('カラン', 'ドンドン'),  # 金属音 → 太鼓のような音
                ('キラキラ', 'ゴロゴロ'),  # 明るい音 → 暗く重い音
            ]
        },
        {
            'name': 'coin',
            'audio_path': '../selected_files/c1/coins1/051.wav',
            'transformations': [
                ('カラン', 'ゴロゴロ'),  # 金属音 → 重厚な音
                ('カラン', 'バシャバシャ'),  # 金属音 → 水の音
                ('チャリン', 'ドンドン'),  # 軽い金属音 → 太鼓の音
                ('カラン', 'サラサラ'),  # 金属音 → 軽やかな音
                ('キラキラ', 'ガンガン'),  # 明るい音 → 激しい音
            ]
        },
        {
            'name': 'wood',
            'audio_path': '../selected_files/a1/wood2/015.wav',
            'transformations': [
                ('カッ', 'ガッ'),  # 清音 → 濁音
                ('コツ', 'ドン'),  # 軽い音 → 重い音
                ('カラカラ', 'ゴロゴロ'),  # 軽やか → 重厚
                ('カッ', 'ドンドン'),  # 短い音 → 連続した重い音
                ('トン', 'ガンガン'),  # 単発 → 激しい連続音
            ]
        },
    ]

    # 2. 出力フォルダ
    output_dir = '../demo_audio/differential_test'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # 3. モデルロード
    print("[2] Loading RWCP model...")
    processor = Onoma2DSP(
        model_path='../models/rwcp_model.pth',
        scaler_path='../models/rwcp_scaler.pkl',
        amplification_factor=5.0
    )
    print("Model loaded!\n")

    # 4. 各変換を実行
    print("[3] Processing transformations...")
    print("=" * 80)

    all_results = []

    for case_idx, case in enumerate(test_cases, 1):
        name = case['name']
        audio_path = case['audio_path']

        if not os.path.exists(audio_path):
            print(f"[SKIP] Audio not found: {audio_path}")
            continue

        print(f"\n{'='*80}")
        print(f"[{case_idx}] Sound: {name.upper()}")
        print(f"    Input: {audio_path}")
        print(f"{'='*80}\n")

        # 元の音声をコピー
        original_output = os.path.join(output_dir, f"{case_idx:02d}_{name}_original.wav")
        shutil.copy(audio_path, original_output)

        # 各変換を実行
        for trans_idx, (source_onoma, target_onoma) in enumerate(case['transformations'], 1):
            output_path = os.path.join(
                output_dir,
                f"{case_idx:02d}_{name}_{trans_idx:02d}_{source_onoma}_to_{target_onoma}.wav"
            )

            print(f"  [{trans_idx}/5] {source_onoma} -> {target_onoma}...", end=' ')

            try:
                processor.process(
                    source_onoma,
                    target_onoma,
                    audio_path,
                    output_path,
                    verbose=False
                )
                print("[OK]")

                all_results.append({
                    'sound': name,
                    'source': source_onoma,
                    'target': target_onoma,
                    'original': original_output,
                    'processed': output_path
                })

            except Exception as e:
                print(f"[ERROR: {e}]")

    # 5. 結果ガイドを作成
    guide_path = os.path.join(output_dir, 'LISTENING_GUIDE.txt')
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("差分ベースオノマトペシステム - 音声変換結果\n")
        f.write("=" * 80 + "\n\n")
        f.write("このフォルダには、2つのオノマトペの差分を使って音声を変換した結果が含まれています。\n")
        f.write("Source（元の音の特徴）→ Target（目標の音の特徴）への変換を試しています。\n\n")

        # 音源ごとにグループ化
        sounds = list(set([r['sound'] for r in all_results]))

        for sound in sounds:
            f.write("=" * 80 + "\n")
            f.write(f"{sound.upper()}\n")
            f.write("=" * 80 + "\n\n")

            sound_results = [r for r in all_results if r['sound'] == sound]
            f.write(f"[元の音] {os.path.basename(sound_results[0]['original'])}\n\n")

            for r in sound_results:
                f.write(f"[{r['source']} -> {r['target']}]\n")
                f.write(f"  {os.path.basename(r['processed'])}\n\n")

        f.write("=" * 80 + "\n")
        f.write("変換のコンセプト\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. 清音 -> 濁音: カッ -> ガッ\n")
        f.write("   軽やかな音 -> 重厚で力強い音\n\n")
        f.write("2. 軽やか -> 重厚: チリン/カラン -> ゴロゴロ\n")
        f.write("   高音で明るい音 -> 低音で重い音\n\n")
        f.write("3. 単発 -> 連続: カッ -> ドンドン/ガンガン\n")
        f.write("   短い音 -> 連続した激しい音\n\n")
        f.write("4. 明るい -> 暗い: キラキラ -> ゴロゴロ/ガンガン\n")
        f.write("   高音強調 -> 低音強調\n\n")

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETED")
    print("=" * 80)
    print(f"\nTotal processed: {len(all_results)} transformations")
    print(f"Output directory: {output_dir}")
    print(f"Listening guide: {guide_path}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
