"""学習データの音声ファイルを確認"""
import pandas as pd
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/rwcp_dataset_audio_based.csv')
print('カラム:', list(df.columns)[:5])
print()

# ユニークなオノマトペとその音声ファイル
print('音声ファイル例（最初の20件）:')
for i, row in df.head(20).iterrows():
    path = row['audio_path']
    exists = os.path.exists(path)
    status = "OK" if exists else "NG"
    onoma = row['onomatopoeia_katakana']
    print(f'  [{status}] {onoma} - {path}')

# 存在する音声ファイルを探す
print('\n\n存在する音声ファイルを探索中...')
existing = []
for i, row in df.iterrows():
    if os.path.exists(row['audio_path']):
        existing.append((row['audio_path'], row['onomatopoeia_katakana']))
    if len(existing) >= 20:
        break

print(f'\n見つかった音声ファイル ({len(existing)}件):')
for path, onoma in existing:
    print(f'  {onoma} - {path}')
