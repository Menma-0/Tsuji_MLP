# Onoma2DSP 使い方ガイド

オノマトペを使って音声を変換するシステムの使用方法を説明します。

## 📋 目次
1. [基本的な使い方](#基本的な使い方)
2. [パラメータ解説](#パラメータ解説)
3. [使用例](#使用例)
4. [Tips & Tricks](#tips--tricks)

---

## 基本的な使い方

### 1. システムの初期化

```python
from src.onoma2dsp import Onoma2DSP

processor = Onoma2DSP(
    model_path='models/rwcp_model.pth',      # 学習済みモデル
    scaler_path='models/rwcp_scaler.pkl',    # スケーラー
    amplification_factor=5.0,                 # DSP効果の強度
    lambda_att=0.7                            # Attention機能の強度
)
```

### 2. 音声の変換

```python
result = processor.process(
    source_onomatopoeia='チリン',    # 現在の音のイメージ
    target_onomatopoeia='ゴロゴロ',  # 目標の音のイメージ
    input_audio_path='input.wav',
    output_audio_path='output.wav',
    verbose=True                      # 処理過程を表示
)
```

### 3. 結果の確認

```python
print(f"出力ファイル: {result['output_path']}")
print(f"特徴量の差分: {result['feature_diff_magnitude']:.2f}")
print(f"適用されたDSPパラメータ: {result['mapped_params']}")
```

---

## パラメータ解説

### 初期化パラメータ

| パラメータ | 型 | 推奨値 | 説明 |
|-----------|-----|--------|------|
| `model_path` | str | 必須 | 学習済みMLPモデルのパス |
| `scaler_path` | str | 必須 | StandardScalerのパス |
| `amplification_factor` | float | 3.0-7.0 | DSP効果の強度。大きいほど変換が劇的に |
| `lambda_att` | float | 0.0-0.7 | Attention機能の強度。0.0=OFF, 0.5-0.7=ON推奨 |
| `sample_rate` | int | 44100 | サンプリングレート（通常は変更不要） |
| `device` | str | 'cpu' | 使用デバイス（'cpu' or 'cuda'） |

### processメソッドのパラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `source_onomatopoeia` | str | 入力音声の印象を表すオノマトペ（カタカナ） |
| `target_onomatopoeia` | str | 目標の印象を表すオノマトペ（カタカナ） |
| `input_audio_path` | str | 入力音声ファイル（.wav） |
| `output_audio_path` | str | 出力音声ファイル（.wav） |
| `verbose` | bool | True=処理過程を表示, False=結果のみ |

---

## 使用例

### 例1: 基本的な変換（Attention OFF）

高音の「チリン」を低音の「ゴロゴロ」に変換：

```python
processor = Onoma2DSP(
    model_path='models/rwcp_model.pth',
    scaler_path='models/rwcp_scaler.pkl',
    amplification_factor=5.0,
    lambda_att=0.0  # Attention OFF
)

result = processor.process(
    source_onomatopoeia='チリン',
    target_onomatopoeia='ゴロゴロ',
    input_audio_path='bell.wav',
    output_audio_path='bell_transformed.wav'
)
```

### 例2: Attention機能を使う

同じ変換をAttention機能ONで実行（より自然な変換）：

```python
processor = Onoma2DSP(
    model_path='models/rwcp_model.pth',
    scaler_path='models/rwcp_scaler.pkl',
    amplification_factor=5.0,
    lambda_att=0.7  # Attention ON
)

result = processor.process(
    source_onomatopoeia='チリン',
    target_onomatopoeia='ゴロゴロ',
    input_audio_path='bell.wav',
    output_audio_path='bell_transformed_att.wav'
)
```

**Attention機能の効果:**
- ソースオノマトペ「チリン」が高音系なので、高音領域の変化が強調されます
- ユーザーの意図（「チリン」のどこを聞いているか）を推定して補正
- より自然で意図に沿った変換が可能

### 例3: 様々な変換パターン

```python
# 軽快 → 重厚
processor.process('コツコツ', 'ドンドン', 'wood.wav', 'wood_heavy.wav')

# 細かい → 粗い
processor.process('サラサラ', 'ザラザラ', 'sand.wav', 'sand_rough.wav')

# 清音 → 濁音
processor.process('カッ', 'ガッ', 'hit.wav', 'hit_voiced.wav')

# 明るい → 激しい
processor.process('キラキラ', 'ガンガン', 'metal.wav', 'metal_intense.wav')
```

### 例4: 強度の調整

```python
# 控えめな変換
processor_weak = Onoma2DSP(
    model_path='models/rwcp_model.pth',
    scaler_path='models/rwcp_scaler.pkl',
    amplification_factor=3.0,  # 弱め
    lambda_att=0.5
)

# 標準的な変換
processor_normal = Onoma2DSP(
    model_path='models/rwcp_model.pth',
    scaler_path='models/rwcp_scaler.pkl',
    amplification_factor=5.0,  # 標準
    lambda_att=0.7
)

# 劇的な変換
processor_strong = Onoma2DSP(
    model_path='models/rwcp_model.pth',
    scaler_path='models/rwcp_scaler.pkl',
    amplification_factor=7.0,  # 強め
    lambda_att=0.7
)
```

---

## Tips & Tricks

### 💡 オノマトペの選び方

**高さ（周波数）:**
- 高音: チリン、キーン、ピンポン、キラキラ
- 低音: ゴロゴロ、ドーン、ガンガン、ズンズン

**音色:**
- 鋭い: カッ、キッ、ピッ
- 鈍い: ガッ、ゴッ、ドッ

**質感:**
- 硬い: コツコツ、カチカチ
- 柔らかい: ポンポン、フワフワ

**粒度:**
- 細かい: サラサラ、シャラシャラ
- 粗い: ザラザラ、ガラガラ

### ⚙️ パラメータ調整のコツ

**amplification_factor:**
- 3.0: 控えめな変換（元の音を保ちつつ調整）
- 5.0: 標準（バランスの取れた変換）
- 7.0: 劇的な変換（大胆な変化）
- それ以上: 不自然になる可能性が高い

**lambda_att:**
- 0.0: Attention OFF（基本的な変換）
- 0.5: 弱めのAttention（控えめに意図を反映）
- 0.7: 標準的なAttention（推奨）
- 0.9: 強めのAttention（変化が劇的になる）

### 🎯 良い結果を得るために

1. **適切なオノマトペを選ぶ**
   - 入力音声の印象に合ったsource_onomatopoeiaを選ぶ
   - 明確に異なるtarget_onomatopoeiaを選ぶ（似すぎると変化が小さい）

2. **音源に応じた調整**
   - 音楽素材: amplification_factor=3.0-4.0（控えめ）
   - 効果音: amplification_factor=5.0-7.0（標準〜強め）

3. **Attention機能を活用**
   - より自然な変換を求める場合はON推奨
   - 実験的な変換をする場合はOFFも試す

4. **結果の評価**
   - 複数のパラメータ設定で試す
   - baseline（lambda_att=0.0）とAttention ON（lambda_att=0.7）を比較
   - 主観的な聴感を重視

---

## 🚀 クイックスタート

最も簡単に試すには、以下のデモスクリプトを実行してください：

```bash
python demo_usage.py
```

このスクリプトは:
- 4つの使用例を実行
- 様々なパラメータ設定を試す
- demo_audio/フォルダに結果を保存

生成された音声ファイルを聴き比べることで、システムの動作を理解できます。

---

## 📝 よくある質問

**Q: どんな音声ファイルが使えますか？**
A: WAV形式の音声ファイルが使用できます。モノラル・ステレオどちらでも対応しています。

**Q: Attention機能は必ず使うべきですか？**
A: 推奨ですが必須ではありません。まずはOFF（0.0）とON（0.7）の両方を試して、好みの方を選んでください。

**Q: オノマトペはどこまで自由に作れますか？**
A: カタカナで表現可能なオノマトペなら基本的に使用できます。ただし、学習データに含まれる音韻パターンに近いほど精度が高くなります。

**Q: 変換結果が不自然な場合は？**
A: amplification_factorを下げる（例: 5.0 → 3.0）、またはより適切なオノマトペを選ぶことを試してください。

---

## 📚 関連ファイル

- `demo_usage.py`: 使用例デモスクリプト
- `src/onoma2dsp.py`: メインシステム
- `test_attention_single.py`: 詳細なテストスクリプト
- `test_attention_comprehensive.py`: 包括的なテストスクリプト

---

**バージョン:** 1.0
**最終更新:** 2025-12-03
