"""ジャッジャッ → タッタッ の編集内容を詳細分析
2つの解釈（差分モデル / Attention）を分けて表示"""
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

# 前処理モジュール
katakana_converter = KatakanaToPhoneme()
mora_converter = PhonemeToMora()
feature_extractor = OnomatopoeiaFeatureExtractor()

source = 'ジャッジャッ'
target = 'タッタッ'

print('=' * 70)
print('【分析対象】')
print('=' * 70)
print(f'  Source（元音声）: {source}')
print(f'  Target（目標音声）: {target}')

# ========================================
# STEP 1: 特徴量の抽出
# ========================================
print()
print('=' * 70)
print('【STEP 1】特徴量の抽出')
print('=' * 70)

# Source
source_phonemes = katakana_converter.convert(source)
source_moras = mora_converter.convert(source_phonemes)
source_features = feature_extractor.extract_features(source_phonemes, source_moras)

print(f'\n  {source}:')
print(f'    音素: {source_phonemes}')
print(f'    モーラ: {["".join(m) for m in source_moras]}')

# Target
target_phonemes = katakana_converter.convert(target)
target_moras = mora_converter.convert(target_phonemes)
target_features = feature_extractor.extract_features(target_phonemes, target_moras)

print(f'\n  {target}:')
print(f'    音素: {target_phonemes}')
print(f'    モーラ: {["".join(m) for m in target_moras]}')

# 差分
feature_diff = target_features - source_features
print(f'\n  特徴量差分ベクトルの大きさ: {np.linalg.norm(feature_diff):.3f}')

# ========================================
# STEP 2: 解釈①「差分モデルの出力」
# ========================================
print()
print('=' * 70)
print('【STEP 2】解釈①: 差分モデルの出力')
print('  → 「ジャッジャッ」と「タッタッ」の特徴量差分をMLPに入力')
print('  → MLPが「どう音を変えるべきか」を10次元で出力')
print('=' * 70)

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

print('\n  【モデルの生出力】（-1〜+1の範囲、Tanhで制限）')
for i, name in enumerate(PARAM_NAMES):
    bar = '█' * int(abs(raw_params[i]) * 20)
    sign = '+' if raw_params[i] >= 0 else '-'
    print(f'    {PARAM_NAMES_JP[name]:<15}: {raw_params[i]:>7.3f}  {sign}{bar}')

# 増幅
amplification_factor = 1.0
amplified_params = raw_params * amplification_factor
clipped_params = np.clip(amplified_params, -1.0, 1.0)

print(f'\n  【増幅処理】（×{amplification_factor}倍して-1〜+1にクリップ）')
print('  理由: モデル出力が小さすぎて変化が聞こえにくいため')
for i, name in enumerate(PARAM_NAMES):
    bar = '█' * int(abs(clipped_params[i]) * 20)
    sign = '+' if clipped_params[i] >= 0 else '-'
    print(f'    {PARAM_NAMES_JP[name]:<15}: {raw_params[i]:>6.3f} × 5 = {amplified_params[i]:>6.3f} → clip → {clipped_params[i]:>6.3f}  {sign}{bar}')

# ========================================
# STEP 3: 解釈②「Attentionベクトル」
# ========================================
print()
print('=' * 70)
print('【STEP 3】解釈②: Attentionベクトル（元音声オノマトペから抽出）')
print(f'  → 「{source}」のDSPテンプレートから「どの帯域に注目すべきか」を計算')
print('  → 絶対値を取り、0〜1に正規化')
print('=' * 70)

# Attentionの計算
template_source = create_dsp_template(source)
temp_array = np.array(template_source)

print(f'\n  【{source} のDSPテンプレート】')
print('  （このオノマトペが本来持つ音響特性の定義）')
for i, name in enumerate(PARAM_NAMES):
    print(f'    {PARAM_NAMES_JP[name]:<15}: {temp_array[i]:>7.3f}')

# 絶対値を取る
attention_raw = np.abs(temp_array)
print(f'\n  【絶対値を取る】')
print('  理由: 値が大きい（正でも負でも）＝その帯域が重要')
for i, name in enumerate(PARAM_NAMES):
    print(f'    {PARAM_NAMES_JP[name]:<15}: |{temp_array[i]:>6.3f}| = {attention_raw[i]:>6.3f}')

# 0〜1に正規化
max_att = np.max(attention_raw)
attention = attention_raw / max_att if max_att > 1e-8 else attention_raw
attention = np.clip(attention, 0.0, 1.0)

print(f'\n  【0〜1に正規化】（最大値 {max_att:.3f} で割る）')
print('  これが「Attentionベクトル」= 各パラメータの重要度')
for i, name in enumerate(PARAM_NAMES):
    bar = '█' * int(attention[i] * 20)
    print(f'    {PARAM_NAMES_JP[name]:<15}: {attention[i]:>6.3f}  {bar}')

# ========================================
# STEP 4: 2つの解釈の統合
# ========================================
print()
print('=' * 70)
print('【STEP 4】2つの解釈の統合')
print('  公式: 最終出力 = 差分モデル出力 × (1 + λ × Attention)')
print(f'  λ (lambda_att) = 0.5')
print('=' * 70)

lambda_att = 0.5

print(f'\n  【計算過程】')
print(f'  差分モデル × (1 + 0.5 × Attention) = 最終出力')
print()

final_params = clipped_params * (1.0 + lambda_att * attention)
final_params = np.clip(final_params, -1.0, 1.0)

for i, name in enumerate(PARAM_NAMES):
    multiplier = 1.0 + lambda_att * attention[i]
    before_clip = clipped_params[i] * multiplier
    print(f'    {PARAM_NAMES_JP[name]:<15}: {clipped_params[i]:>6.3f} × (1 + 0.5 × {attention[i]:.3f}) = {clipped_params[i]:>6.3f} × {multiplier:.3f} = {before_clip:>6.3f} → {final_params[i]:>6.3f}')

# ========================================
# STEP 5: 最終的なDSPパラメータ
# ========================================
print()
print('=' * 70)
print('【STEP 5】最終的なDSPパラメータ（実際に音声に適用される値）')
print('=' * 70)

mapper = DSPParameterMapping()
mapped_params = mapper.map_parameters(final_params)

print('\n  【マッピング後の値】')
for key, value in mapped_params.items():
    if 'db' in key:
        print(f'    {key:<25}: {value:>8.2f} dB')
    elif 'ratio' in key:
        print(f'    {key:<25}: {value:>8.2f} x')
    else:
        print(f'    {key:<25}: {value:>8.2f}')

# ========================================
# STEP 6: 編集の解釈まとめ
# ========================================
print()
print('=' * 70)
print('【STEP 6】編集の解釈まとめ')
print('=' * 70)

print('\n  ■ 解釈①（差分モデル）が言っていること:')
print('    「ジャッジャッ → タッタッ」の変化は...')
interpretation1 = []
if clipped_params[0] > 0.3:
    interpretation1.append('    → 音量を上げる')
if clipped_params[1] > 0.3:
    interpretation1.append('    → 圧縮を強める')
if clipped_params[2] > 0.3 or clipped_params[3] > 0.3:
    interpretation1.append('    → 低音を強調する')
if clipped_params[4] < -0.3:
    interpretation1.append('    → 中音をカットする')
if clipped_params[5] < -0.3 or clipped_params[6] < -0.3:
    interpretation1.append('    → 高音をカットする')
if clipped_params[7] > 0.3:
    interpretation1.append('    → アタックを強調する')
if clipped_params[8] > 0.3:
    interpretation1.append('    → サステインを長くする')
if clipped_params[9] > 0.3:
    interpretation1.append('    → 音を遅く（長く）する')
for line in interpretation1:
    print(line)

print('\n  ■ 解釈②（Attention）が言っていること:')
print(f'    「{source}」という音は...')
# 上位3つのAttention
att_sorted = sorted(enumerate(attention), key=lambda x: x[1], reverse=True)
for idx, val in att_sorted[:5]:
    if val > 0.3:
        print(f'    → {PARAM_NAMES_JP[PARAM_NAMES[idx]]} が重要（注目度: {val:.2f}）')

print('\n  ■ 統合結果:')
print('    Attentionが高い次元は、差分モデルの出力がさらに強調される')
print('    → 元音声の特徴的な帯域での変化がより大きくなる')

# 実際の効果
print('\n  ■ 実際に起きた編集:')
if mapped_params['gain_db'] > 3:
    print(f'    ・音量が +{mapped_params["gain_db"]:.1f}dB 大きくなった')
if mapped_params['eq_sub_db'] > 6 or mapped_params['eq_low_db'] > 6:
    print(f'    ・低音が大幅に強調された（重い音に）')
if mapped_params['eq_high_db'] < -6 or mapped_params['eq_presence_db'] < -6:
    print(f'    ・高音が大幅にカットされた（こもった音に）')
if mapped_params['transient_attack'] > 0.5:
    print(f'    ・アタックが強調された（鋭い立ち上がりに）')
if mapped_params['time_stretch_ratio'] > 1.2:
    print(f'    ・音が {(mapped_params["time_stretch_ratio"]-1)*100:.0f}% 遅くなった（長くなった）')
