"""ジャッジャッ → タッタッ の編集内容を分析"""
import sys
sys.path.append('src')

from preprocessing.katakana_to_phoneme import KatakanaToPhoneme
from preprocessing.phoneme_to_mora import PhonemeToMora
from preprocessing.feature_extractor import OnomatopoeiaFeatureExtractor
from models.mlp_model import Onoma2DSPMLP, DSPParameterMapping
from utils.create_rwcp_dataset import create_dsp_template
import torch
import pickle
import numpy as np

# 前処理モジュール
katakana_converter = KatakanaToPhoneme()
mora_converter = PhonemeToMora()
feature_extractor = OnomatopoeiaFeatureExtractor()

source = 'ジャッジャッ'
target = 'タッタッ'

print('='*60)
print('Source:', source)
print('Target:', target)
print('='*60)

# Source
source_phonemes = katakana_converter.convert(source)
source_moras = mora_converter.convert(source_phonemes)
source_features = feature_extractor.extract_features(source_phonemes, source_moras)

print()
print('【Source:', source, '】')
print('  音素:', source_phonemes)
print('  モーラ:', [''.join(m) for m in source_moras])

# Target
target_phonemes = katakana_converter.convert(target)
target_moras = mora_converter.convert(target_phonemes)
target_features = feature_extractor.extract_features(target_phonemes, target_moras)

print()
print('【Target:', target, '】')
print('  音素:', target_phonemes)
print('  モーラ:', [''.join(m) for m in target_moras])

# 特徴量の差分
feature_diff = target_features - source_features
print()
print('【特徴量の差分】')
print('  差分の大きさ:', round(np.linalg.norm(feature_diff), 3))

# モデルで予測
model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=32, use_tanh=True)
model.load_state_dict(torch.load('models/saved_model.pth', map_location='cpu'))
model.eval()

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

feature_diff_scaled = scaler.transform(feature_diff.reshape(1, -1))[0]

with torch.no_grad():
    diff_tensor = torch.FloatTensor(feature_diff_scaled).unsqueeze(0)
    raw_params = model(diff_tensor).cpu().numpy()[0]

# 増幅（デフォルト5倍）
amplification_factor = 1.0
normalized_params = np.clip(raw_params * amplification_factor, -1.0, 1.0)

# Attention補正
lambda_att = 0.5
template_source = create_dsp_template(source)
temp_array = np.array(template_source)
attention = np.abs(temp_array)
max_att = np.max(attention)
if max_att > 1e-8:
    attention = attention / max_att
attention = np.clip(attention, 0.0, 1.0)

params_with_attention = normalized_params * (1.0 + lambda_att * attention)
params_with_attention = np.clip(params_with_attention, -1.0, 1.0)

# パラメータをマッピング
mapper = DSPParameterMapping()
mapped_params = mapper.map_parameters(params_with_attention)

print()
print('【モデルの生出力 → 増幅(x5) → attention補正】')
param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid', 'eq_high', 'eq_presence', 'attack', 'sustain', 'stretch']
for i, name in enumerate(param_names):
    print('  %-15s: %7.3f -> %7.3f -> %7.3f' % (name, raw_params[i], normalized_params[i], params_with_attention[i]))

print()
print('='*60)
print('【実際に適用されたDSPパラメータ】')
print('='*60)
for key, value in mapped_params.items():
    if 'db' in key:
        print('  %-25s: %8.2f dB' % (key, value))
    elif 'ratio' in key:
        print('  %-25s: %8.2f x' % (key, value))
    else:
        print('  %-25s: %8.2f' % (key, value))

print()
print('='*60)
print('【編集の意味（日本語で解説）】')
print('='*60)

# 解説を生成
explanations = []

gain = mapped_params['gain_db']
if gain > 2:
    explanations.append('音量を %.1f dB 上げた（大きくなった）' % gain)
elif gain < -2:
    explanations.append('音量を %.1f dB 下げた（小さくなった）' % abs(gain))

eq_sub = mapped_params['eq_sub_db']
eq_low = mapped_params['eq_low_db']
if eq_sub > 3 or eq_low > 3:
    explanations.append('低音を強調した（重い音に）')
elif eq_sub < -3 or eq_low < -3:
    explanations.append('低音をカットした（軽い音に）')

eq_high = mapped_params['eq_high_db']
eq_presence = mapped_params['eq_presence_db']
if eq_high > 3 or eq_presence > 3:
    explanations.append('高音を強調した（明るい・シャープな音に）')
elif eq_high < -3 or eq_presence < -3:
    explanations.append('高音をカットした（落ち着いた音に）')

attack = mapped_params['transient_attack']
if attack > 0.3:
    explanations.append('アタック（立ち上がり）を強調した（鋭い音に）')
elif attack < -0.3:
    explanations.append('アタック（立ち上がり）を弱めた（柔らかい音に）')

sustain = mapped_params['transient_sustain']
if sustain > 0.3:
    explanations.append('サステイン（持続）を長くした')
elif sustain < -0.3:
    explanations.append('サステイン（持続）を短くした')

stretch = mapped_params['time_stretch_ratio']
if stretch > 1.1:
    explanations.append('音を %.0f%% 遅くした（長くなった）' % ((stretch - 1) * 100))
elif stretch < 0.9:
    explanations.append('音を %.0f%% 速くした（短くなった）' % ((1 - stretch) * 100))

if explanations:
    for exp in explanations:
        print('  ・' + exp)
else:
    print('  ・変化が小さく、ほぼ元の音のまま')
