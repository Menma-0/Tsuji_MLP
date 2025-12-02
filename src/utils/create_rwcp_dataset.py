"""
RWCP-SSD-OnomatopoeiaデータからMLPモデル用のデータセットを作成
"""
import pandas as pd
import numpy as np
import os
import sys
import librosa

# プロジェクトルートを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.preprocessing.katakana_to_phoneme import KatakanaToPhoneme
from src.preprocessing.phoneme_to_mora import PhonemeToMora
from src.preprocessing.feature_extractor import OnomatopoeiaFeatureExtractor


def create_dsp_template(onomatopoeia):
    """
    オノマトペからDSPパラメータのテンプレートを生成

    ヒューリスティックなルール:
    - 濁音（ガ、ザ、ダ、バなど）→ 低音強調、音量大
    - 清音高音（キ、ピ、チなど）→ 高音強調、音量小
    - 促音（ッ）→ アタック強化
    - 長音（ー）→ サスティン延長、タイムストレッチ
    - 繰り返し → 音量増加
    """

    # 音素変換
    k2p = KatakanaToPhoneme()
    p2m = PhonemeToMora()

    phonemes = k2p.convert(onomatopoeia)
    moras = p2m.convert(phonemes)

    # 初期値（ニュートラル）
    params = [0.0] * 10  # [gain, compression, eq_sub, eq_low, eq_mid, eq_high, eq_presence, attack, sustain, time_stretch]

    # 濁音カウント（g, d, z, b）
    voiced_count = sum(1 for p in phonemes if p in ['g', 'd', 'z', 'b'])
    # 高音系子音（k, p, t, s, sh, ch, ts）
    high_consonants = sum(1 for p in phonemes if p in ['k', 'p', 't', 's', 'sh', 'ch', 'ts'])
    # 促音
    sokuon_count = phonemes.count('Q')
    # 長音
    choon_count = phonemes.count('H')
    # モーラ数
    mora_count = len(moras)

    # ルール適用（より感度を高くして、より多くのパターンに反応）

    # 1. Gain（音量）
    if voiced_count >= 2:
        params[0] = 0.3 + 0.2 * min(voiced_count, 4)  # 濁音 → 大音量
    elif high_consonants >= 1:
        params[0] = -0.2 - 0.1 * min(high_consonants, 3)  # 高音系 → 控えめ

    # 2. Compression（圧縮）
    if voiced_count >= 2:
        params[1] = 0.3 + 0.1 * min(voiced_count, 4)
    elif sokuon_count > 0:
        params[1] = 0.3

    # 3. EQ Sub（超低域 80Hz）
    if voiced_count >= 2:
        params[2] = 0.4 + 0.2 * min(voiced_count, 4)  # 濁音 → 低音強化
    elif high_consonants >= 1:
        params[2] = -0.3 - 0.1 * min(high_consonants, 3)  # 高音系 → 低音カット

    # 4. EQ Low（低域 250Hz）
    if voiced_count >= 1:
        params[3] = 0.3 + 0.2 * min(voiced_count, 4)
    elif high_consonants >= 1:
        params[3] = -0.2 - 0.1 * min(high_consonants, 3)

    # 5. EQ Mid（中域 1kHz）
    params[4] = 0.0  # ニュートラル

    # 6. EQ High（高域 4kHz）
    if high_consonants >= 1:
        params[5] = 0.4 + 0.2 * min(high_consonants, 4)  # 高音系 → 高音強調
    elif voiced_count >= 2:
        params[5] = -0.2 - 0.1 * min(voiced_count, 3)  # 濁音 → 高音カット

    # 7. EQ Presence（超高域 10kHz）
    if high_consonants >= 1:
        params[6] = 0.5 + 0.2 * min(high_consonants, 4)  # さらに強調
    elif voiced_count >= 2:
        params[6] = -0.2 - 0.1 * min(voiced_count, 3)

    # 8. Transient Attack（アタック）
    if sokuon_count > 0:
        params[7] = 0.7  # 促音 → 鋭いアタック
    elif choon_count > 1:
        params[7] = -0.4  # 長音 → 柔らかいアタック
    else:
        params[7] = 0.0

    # 9. Transient Sustain（サスティン）
    if choon_count > 1:
        params[8] = 0.6  # 長音 → 長いサスティン
    elif sokuon_count > 0:
        params[8] = -0.3  # 促音 → 短い

    # 10. Time Stretch（時間伸縮）
    if choon_count > 2:
        params[9] = 0.3  # 長音多い → やや伸ばす
    elif sokuon_count > 1:
        params[9] = -0.2  # 促音多い → やや短く

    return params


def main():
    print("=" * 80)
    print("RWCP-SSD-Onomatopoeia Dataset Creation")
    print("=" * 80)

    # 1. CSVを読み込み
    csv_path = 'training_data_jp_utf8bom.csv'
    print(f"\nLoading CSV: {csv_path}")

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"Total samples loaded: {len(df)}")
    print(f"Unique onomatopoeia: {df['onomatopoeia'].nunique()}")

    # 2. 高スコアのサンプルのみフィルタ（confidence >= 4 かつ avg_acceptability >= 4）
    print("\nFiltering high-quality samples (confidence >= 4 and avg_acceptability >= 4)...")
    df_filtered = df[(df['confidence'] >= 4) & (df['avg_acceptability'] >= 4.0)]
    print(f"Samples after filtering: {len(df_filtered)}")

    # 3. audio_pathを実際のパスに変換
    # RWCP-SSD/drysrc/a1/cherry1/043.wav → selected_files/a1/cherry1/043.wav
    df_filtered['audio_path'] = df_filtered['audio_path'].str.replace('RWCP-SSD/drysrc/', 'selected_files/')

    # 4. ファイルの存在確認
    print("\nChecking file existence...")
    existing_files = []
    for idx, row in df_filtered.iterrows():
        if os.path.exists(row['audio_path']):
            existing_files.append(idx)

    df_existing = df_filtered.loc[existing_files]
    print(f"Existing audio files: {len(df_existing)}")

    if len(df_existing) == 0:
        print("ERROR: No audio files found! Please check the audio_path mapping.")
        return

    # 5. データセット作成
    print("\nCreating dataset...")

    dataset_rows = []

    for idx, row in df_existing.iterrows():
        onomatopoeia = row['onomatopoeia']
        audio_path = row['audio_path']

        # sound_idを生成（ファイルパスから）
        # selected_files/a1/cherry1/043.wav → a1_cherry1_043
        sound_id = audio_path.replace('selected_files/', '').replace('/', '_').replace('.wav', '')

        # DSPパラメータを生成
        dsp_params = create_dsp_template(onomatopoeia)

        # 行を作成
        row_data = {
            'sound_id': sound_id,
            'onomatopoeia_katakana': onomatopoeia,
            'confidence_score': row['confidence'],
            'acceptance_score': row['avg_acceptability']
        }

        # DSPパラメータを追加
        for i, param in enumerate(dsp_params):
            row_data[f'dsp_target_{i}'] = param

        dataset_rows.append(row_data)

        if (len(dataset_rows) % 1000) == 0:
            print(f"  Processed: {len(dataset_rows)} samples")

    # 6. DataFrameに変換
    dataset_df = pd.DataFrame(dataset_rows)

    # 7. 保存
    output_path = 'data/rwcp_dataset.csv'
    os.makedirs('data', exist_ok=True)
    dataset_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*80}")
    print(f"Dataset saved to: {output_path}")
    print(f"Total samples: {len(dataset_df)}")
    print(f"Unique onomatopoeia: {dataset_df['onomatopoeia_katakana'].nunique()}")
    print(f"{'='*80}")

    # 統計情報
    print("\nTop 10 onomatopoeia:")
    print(dataset_df['onomatopoeia_katakana'].value_counts().head(10))

    print("\nDSP parameter statistics:")
    dsp_cols = [f'dsp_target_{i}' for i in range(10)]
    print(dataset_df[dsp_cols].describe())


if __name__ == '__main__':
    main()
