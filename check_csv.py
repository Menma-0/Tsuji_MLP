import pandas as pd

df = pd.read_csv('training_data_jp_utf8bom.csv', encoding='utf-8-sig')
print('Total rows:', len(df))
print('\nFirst 5 rows:')
for idx, row in df.head(5).iterrows():
    print(f"  {idx}: {row['onomatopoeia']} | {row['audio_path']}")

print('\nChecking file existence:')
import os
for idx, row in df.head(5).iterrows():
    path = row['audio_path'].replace('RWCP-SSD/drysrc/', 'selected_files/')
    exists = os.path.exists(path)
    print(f"  {path}: {exists}")
