import pandas as pd
import os

csv_path = 'training_data_jp_utf8bom.csv'
df = pd.read_csv(csv_path, encoding='utf-8-sig')

print(f'Total rows: {len(df)}')
print(f'\nColumns: {df.columns.tolist()}')

# カテゴリ検索テスト
categories = [
    'a1/cherry', 'a1/magno', 'a1/teak', 'a1/wood',
    'b1/particl', 'b2/particl',
    'c1/bell', 'c1/coin', 'c2/glass', 'c3/key'
]

for category in categories:
    category_samples = df[df['audio_path'].str.contains(category)]
    print(f'\n{category}: {len(category_samples)} samples')

    if len(category_samples) > 0:
        sample = category_samples.iloc[0]
        audio_path = sample['audio_path'].replace('RWCP-SSD/drysrc/', 'selected_files/')
        exists = os.path.exists(audio_path)
        print(f'  First: {sample["audio_path"]}')
        print(f'  Converted: {audio_path}')
        print(f'  Exists: {exists}')
