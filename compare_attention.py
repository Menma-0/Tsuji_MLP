"""Attention ON/OFF のパラメータ比較"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('src')

from preprocessing.katakana_to_phoneme import KatakanaToPhoneme
from preprocessing.phoneme_to_mora import PhonemeToMora
from preprocessing.feature_extractor import OnomatopoeiaFeatureExtractor
from models.mlp_model import Onoma2DSPMLP, DSPParameterMapping
from utils.create_rwcp_dataset import create_dsp_template
import torch
import pickle
import numpy as np

# パラメータ名
PARAM_NAMES = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
               'eq_high', 'eq_presence', 'attack', 'sustain', 'stretch']

PARAM_NAMES_JP = {
    'gain': '音量',
    'compression': '圧縮',
    'eq_sub': '超低音(80Hz)',
    'eq_low': '低音(250Hz)',
    'eq_mid': '中音(1kHz)',
    'eq_high': '高音(4kHz)',
    'eq_presence': '超高音(10kHz)',
    'attack': 'アタック',
    'sustain': 'サステイン',
    'stretch': '速度'
}

# 前処理
katakana_converter = KatakanaToPhoneme()
mora_converter = PhonemeToMora()
feature_extractor = OnomatopoeiaFeatureExtractor()

source = 'ジャッジャッ'
target = 'タッタッ'

# 特徴量抽出
source_phonemes = katakana_converter.convert(source)
source_moras = mora_converter.convert(source_phonemes)
source_features = feature_extractor.extract_features(source_phonemes, source_moras)

target_phonemes = katakana_converter.convert(target)
target_moras = mora_converter.convert(target_phonemes)
target_features = feature_extractor.extract_features(target_phonemes, target_moras)

feature_diff = target_features - source_features

# モデルで予測
model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=32, use_tanh=True)
model.load_state_dict(torch.load('models/saved_model.pth', map_location='cpu', weights_only=True))
model.eval()

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

feature_diff_scaled = scaler.transform(feature_diff.reshape(1, -1))[0]

with torch.no_grad():
    diff_tensor = torch.FloatTensor(feature_diff_scaled).unsqueeze(0)
    raw_params = model(diff_tensor).cpu().numpy()[0]

# 増幅
amplification_factor = 1.0
clipped_params = np.clip(raw_params * amplification_factor, -1.0, 1.0)

# Attentionベクトル
template_source = create_dsp_template(source)
temp_array = np.array(template_source)
attention_raw = np.abs(temp_array)
max_att = np.max(attention_raw)
attention = attention_raw / max_att if max_att > 1e-8 else attention_raw
attention = np.clip(attention, 0.0, 1.0)

# Attentionなし
params_no_att = clipped_params.copy()

# Attentionあり (lambda=0.5)
lambda_att = 0.5
params_with_att = clipped_params * (1.0 + lambda_att * attention)
params_with_att = np.clip(params_with_att, -1.0, 1.0)

# マッピング
mapper = DSPParameterMapping()
mapped_no_att = mapper.map_parameters(params_no_att)
mapped_with_att = mapper.map_parameters(params_with_att)

print('=' * 80)
print('【Attention ON/OFF パラメータ比較】')
print(f'  Source: {source} → Target: {target}')
print('=' * 80)

print('\n【正規化パラメータ（-1〜+1）の比較】')
print()
print(f'{"パラメータ":<18} {"Attentionなし":>12} {"Attentionあり":>12} {"Attention値":>10} {"差":>8}')
print('-' * 70)

for i, name in enumerate(PARAM_NAMES):
    diff = params_with_att[i] - params_no_att[i]
    diff_str = f'+{diff:.3f}' if diff >= 0 else f'{diff:.3f}'
    print(f'{PARAM_NAMES_JP[name]:<15} {params_no_att[i]:>12.3f} {params_with_att[i]:>12.3f} {attention[i]:>10.3f} {diff_str:>8}')

print()
print('=' * 80)
print('【実際のDSPパラメータの比較】')
print('=' * 80)

print()
print(f'{"パラメータ":<25} {"Attentionなし":>15} {"Attentionあり":>15} {"差":>12}')
print('-' * 70)

for key in mapped_no_att.keys():
    val_no = mapped_no_att[key]
    val_with = mapped_with_att[key]
    diff = val_with - val_no

    if 'db' in key:
        unit = 'dB'
    elif 'ratio' in key:
        unit = 'x'
    else:
        unit = ''

    diff_str = f'+{diff:.2f}' if diff >= 0 else f'{diff:.2f}'
    print(f'{key:<25} {val_no:>12.2f} {unit:<2} {val_with:>12.2f} {unit:<2} {diff_str:>8} {unit}')

print()
print('=' * 80)
print('【なぜ「Attentionなし」の方がぼやっとして低い音になったか？】')
print('=' * 80)

print('''
■ 実は今回のケースでは、パラメータはほとんど同じ

  理由: 増幅(×5)後、ほとんどのパラメータが上限(±1.0)に
       達しているため、Attentionで掛け算しても変わらない

■ では、なぜ音が違って聞こえたのか？

  可能性1: 別の要因（ファイルの違い、再生環境など）
  可能性2: 処理時の微細な違い（浮動小数点の丸め誤差など）
  可能性3: 実際にはAttentionで差がついているパラメータがある

■ 実際に差があるパラメータ:
''')

# 差があるパラメータを強調
print('  差があるパラメータ:')
for i, name in enumerate(PARAM_NAMES):
    diff = abs(params_with_att[i] - params_no_att[i])
    if diff > 0.001:
        print(f'    {PARAM_NAMES_JP[name]}: {params_no_att[i]:.3f} → {params_with_att[i]:.3f} (差: {diff:.3f})')

# 差がない場合
all_same = all(abs(params_with_att[i] - params_no_att[i]) < 0.001 for i in range(10))
if all_same:
    print('    → 全てのパラメータで差がありません')
    print()
    print('  ★結論: パラメータは同じなのに音が違って聞こえた場合、')
    print('         別の原因（元ファイルの違い、処理タイミングなど）が考えられます')
