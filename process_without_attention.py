"""Attention補正なしで音声を編集するスクリプト"""
import sys
sys.path.append('src')

from onoma2dsp import Onoma2DSP

# === ここを編集 ===
source = 'ジャッジャッ'  # 元音声を表すオノマトペ
target = 'タッタッ'      # 変えたい音を表すオノマトペ
input_file = 'demo_audio/test_walk.wav'   # 入力ファイル
output_file = 'demo_audio/Otest_walk_no_attention.wav'  # 出力ファイル
# ==================

# Attention補正なし（lambda_att=0）で処理
processor = Onoma2DSP(
    model_path='models/saved_model.pth',
    scaler_path='models/scaler.pkl',
    amplification_factor=1.0,  # 増幅率
    lambda_att=0.0             # ★ここが0でAttentionオフ
)

result = processor.process(
    source_onomatopoeia=source,
    target_onomatopoeia=target,
    input_audio_path=input_file,
    output_audio_path=output_file,
    verbose=True
)

print('\n処理完了！')
print(f'出力ファイル: {output_file}')
