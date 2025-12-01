"""
多様なオノマトペを見つける
"""
import pandas as pd
import re

df = pd.read_csv('training_data_jp_utf8bom.csv', encoding='utf-8-sig')

print("Total rows:", len(df))
print("\n=== Looking for diverse onomatopoeia ===\n")

# パターン別に検索
patterns = {
    '濁音（ガ、ザ、ダ、バ）': r'[ガザダバゴゾドボギジヂビグズヅブゲゼデベ]',
    '促音（ッ）': r'ッ',
    '長音（ー）': r'ー',
    '繰り返し4文字以上': r'^(.{2})\1+$',  # サラサラ、ガンガンなど
}

results = {}

for pattern_name, pattern in patterns.items():
    matches = df[df['onomatopoeia'].str.contains(pattern, na=False, regex=True)]
    print(f"\n{pattern_name}: {len(matches)} samples")

    # Top 10を表示
    top_onomas = matches['onomatopoeia'].value_counts().head(10)
    for onoma, count in top_onomas.items():
        # ファイルパスも取得
        sample = matches[matches['onomatopoeia'] == onoma].iloc[0]
        audio_path = sample['audio_path'].replace('RWCP-SSD/drysrc/', 'selected_files/')
        print(f"  {onoma}: {count} samples | {audio_path}")

    results[pattern_name] = matches

print("\n=== 具体的な選択例 ===\n")

# 手動で多様なオノマトペを選択
diverse_onomas = [
    'ガンガン',  # 濁音 + 繰り返し
    'ドンドン',  # 濁音 + 繰り返し
    'バシャバシャ',  # 濁音 + 繰り返し
    'サラサラ',  # 繰り返し
    'キラキラ',  # 繰り返し
    'ガッシャン',  # 濁音 + 促音
    'カッ',  # 促音
    'ピーッ',  # 長音 + 促音
    'ブーッ',  # 濁音 + 長音 + 促音
    'チリーン',  # 長音
]

print("推奨オノマトペ:")
for onoma in diverse_onomas:
    matches = df[df['onomatopoeia'] == onoma]
    if len(matches) > 0:
        sample = matches.iloc[0]
        audio_path = sample['audio_path'].replace('RWCP-SSD/drysrc/', 'selected_files/')
        print(f"  ✓ {onoma}: {len(matches)} samples | {audio_path}")
    else:
        print(f"  ✗ {onoma}: Not found in CSV")
