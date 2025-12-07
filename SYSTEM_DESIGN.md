# Onoma2DSP システム設計書

差分オノマトペによる音声変換システムの技術詳細

**バージョン:** 1.0
**最終更新:** 2025-12-03

---

## 📋 目次

1. [システム概要](#システム概要)
2. [全体アーキテクチャ](#全体アーキテクチャ)
3. [オノマトペ特徴量抽出の詳細](#オノマトペ特徴量抽出の詳細)
4. [差分モデルの詳細](#差分モデルの詳細)
5. [DSPパラメータマッピング](#dspパラメータマッピング)
6. [Attention機構](#attention機構)
7. [学習データとプロセス](#学習データとプロセス)
8. [実装の詳細](#実装の詳細)

---

## システム概要

### コアコンセプト

このシステムは**差分ベース**のオノマトペ音声変換を実現します：

```
入力: source_onomatopoeia（現在の音）+ target_onomatopoeia（目標の音）
      ↓
特徴量差分: Δφ = φ(target) - φ(source)
      ↓
MLPモデル: Δφ → ΔDSP (10次元)
      ↓
出力: 実際のDSPパラメータ
```

**重要な設計思想:**
- オノマトペ自体から**直接**DSPパラメータを予測するのではなく、**2つのオノマトペの差分**からDSP変化を予測
- これにより、「チリン→ゴロゴロ」のような相対的な音質変化をモデル化
- ユーザーは入力音声の特性（source）と目標（target）を両方指定

### システムの特徴

1. **差分アプローチ**: 2つのオノマトペの特徴量差分を計算
2. **軽量MLP**: 38次元入力 → 32次元隠れ層 → 10次元出力
3. **Attention機構**: ソースオノマトペに基づく適応的な補正
4. **履歴記録**: 全ての処理を自動記録

---

## 全体アーキテクチャ

### データフロー

```
[ユーザー入力]
    ├─ source_onomatopoeia: "チリン"
    ├─ target_onomatopoeia: "ゴロゴロ"
    └─ input_audio: bell.wav
         ↓
[1. オノマトペ前処理]
    ├─ カタカナ → 音素列 (KatakanaToPhoneme)
    │   "チリン" → ['ch', 'i', 'r', 'i', 'N']
    ├─ 音素列 → モーラ列 (PhonemeToMora)
    │   → [('ch', 'i'), ('r', 'i'), ('N',)]
    └─ 特徴量抽出 (OnomatopoeiaFeatureExtractor)
        → φ(source): 38次元ベクトル
        → φ(target): 38次元ベクトル
         ↓
[2. 差分計算]
    Δφ = φ(target) - φ(source)  # 38次元
         ↓
[3. 標準化] (Optional)
    Δφ_scaled = StandardScaler.transform(Δφ)
         ↓
[4. MLPモデル推論]
    ΔDSP_norm = MLP(Δφ_scaled)  # 10次元、範囲[-1, +1]
         ↓
[5. Amplification] (Optional)
    ΔDSP_norm = ΔDSP_norm × amplification_factor
    （デフォルト: 1.0 = 等倍）
         ↓
[6. Attention補正] (Optional)
    attention = |create_dsp_template(source)|を正規化
    ΔDSP_final = ΔDSP_norm × (1.0 + lambda_att × attention)
         ↓
[7. DSPパラメータマッピング]
    ΔDSP_final → 実際のdB値、倍率など
    例: gain_db = 24.0 × ΔDSP_final[0]
         ↓
[8. 音声処理]
    ├─ EQ (5バンド)
    ├─ Compression
    ├─ Transient Shaping
    ├─ Time Stretch
    └─ Gain調整
         ↓
[出力音声]
    output.wav
```

### モジュール構成

```
src/
├── preprocessing/
│   ├── katakana_to_phoneme.py      # カタカナ→音素変換
│   ├── phoneme_to_mora.py          # 音素→モーラ変換
│   └── feature_extractor.py        # 特徴量抽出（38次元）
├── models/
│   └── mlp_model.py                # MLPモデル（差分→DSP）
├── dsp/
│   └── dsp_engine.py               # 音声処理エンジン
├── utils/
│   └── create_rwcp_dataset.py      # DSPテンプレート生成
├── onoma2dsp.py                     # メインシステム
├── cli.py                           # CLIインターフェース
└── train_with_rwcp.py               # 学習スクリプト
```

---

## オノマトペ特徴量抽出の詳細

### 3.1 概要

オノマトペ文字列から**38次元の音響的特徴量**を抽出します。

```python
"チリン"
  → ['ch', 'i', 'r', 'i', 'N'] (音素列)
  → [('ch','i'), ('r','i'), ('N',)] (モーラ列)
  → [3.0, 2.0, 2.0, ...] (38次元特徴量)
```

### 3.2 38次元特徴量の内訳

| グループ | 次元数 | 特徴量 | 説明 |
|---------|--------|--------|------|
| **A: 全体構造** | 6 | M, C_count, V_count, word_repeat_count, mora_repeat_chunk_count, mora_repeat_ratio | モーラ数、子音/母音数、繰り返し |
| **B: 長さ・アクセント** | 4 | Q_count, H_mora_count, H_ratio, ending_is_long | 促音、長音の情報 |
| **C: 母音ヒストグラム** | 5 | v_a, v_i, v_u, v_e, v_o | 各母音の出現回数 |
| **D: 子音カテゴリ** | 6 | voiceless_plosive, voiced_plosive, voiceless_fric, voiced_fric, nasal, approximant | 子音の音韻的カテゴリ |
| **E: 子音比率** | 3 | obstruent_ratio, voiced_cons_ratio, nasal_ratio | 子音の性質の比率 |
| **F: 位置情報** | 14 | 語頭/語末の子音カテゴリ（各6次元）、starts/ends_with_vowel | 語頭・語末の音韻情報 |

### 3.3 各グループの詳細

#### グループA: 全体構造・繰り返し（6次元）

```python
def _extract_structure_features(phonemes, moras):
    M = len(moras)  # モーラ数

    # 子音・母音のカウント
    C_count = count_consonants(phonemes)  # 'ch', 'r' など
    V_count = count_vowels(phonemes)      # 'a', 'i', 'u', 'e', 'o'

    # 繰り返しパターン
    word_repeat_count = detect_word_repeat(moras)
    # 例: "ゴロゴロ" → ['go', 'ro', 'go', 'ro'] → 2回繰り返し

    mora_repeat_chunk_count = count_repeat_chunks(moras)
    # 例: "カッカッ" → ['ka', 'Q', 'ka', 'Q'] → 2塊

    mora_repeat_ratio = repeated_moras / M
    # 繰り返しているモーラの割合

    return [M, C_count, V_count, word_repeat_count,
            mora_repeat_chunk_count, mora_repeat_ratio]
```

**音響的解釈:**
- `M`（モーラ数）: 音の長さ・持続時間に対応
- `word_repeat_count`: 周期性・リズム性（ガンガン = 2回）
- `mora_repeat_ratio`: 繰り返しの強さ（リズミカルさ）

#### グループB: 長さ・アクセント（4次元）

```python
def _extract_length_features(phonemes, moras):
    Q_count = count_Q(phonemes)  # 促音「ッ」
    # 例: "カッ" → ['k', 'a', 'Q'] → 1

    H_mora_count = count_long_vowels(moras)  # 長音「ー」
    # 例: "キーン" → ['ki', 'H', 'N'] → 1

    H_ratio = H_mora_count / M

    ending_is_long = 1.0 if moras[-1]に'H'含む else 0.0

    return [Q_count, H_mora_count, H_ratio, ending_is_long]
```

**音響的解釈:**
- `Q_count`: アタックの鋭さ（促音 = 短い無音 = 鋭いアタック）
- `H_mora_count`: サスティン・持続性（長音 = 伸びる音）
- `ending_is_long`: 音の終わり方（減衰特性）

#### グループC: 母音ヒストグラム（5次元）

```python
def _extract_vowel_histogram(phonemes):
    # 各母音の出現回数をカウント
    return [
        count('a'),  # "ア" - 開口度大、明るい
        count('i'),  # "イ" - 高音、鋭い
        count('u'),  # "ウ" - 丸み、こもった
        count('e'),  # "エ" - 中間
        count('o')   # "オ" - 低音、暗い
    ]
```

**音響的解釈:**
- 母音の種類は**フォルマント（共鳴周波数）**に対応
- `i`が多い → 高周波成分が強い（キラキラ、チリン）
- `o`が多い → 低周波成分が強い（ゴロゴロ、ドーン）
- `u`が多い → 中域が弱い、こもった音（ブーン、ズーン）

#### グループD: 子音カテゴリ・ヒストグラム（6次元）

```python
# 子音の音韻的分類
consonant_categories = {
    'voiceless_plosive': ['p', 't', 'k'],  # 無声破裂音
    'voiced_plosive': ['b', 'd', 'g'],     # 有声破裂音
    'voiceless_fric': ['s', 'sh', 'f', 'h'], # 無声摩擦音
    'voiced_fric': ['z', 'j'],             # 有声摩擦音
    'nasal': ['m', 'n', 'N'],              # 鼻音
    'approximant': ['r', 'w', 'y']         # 接近音
}

def _extract_consonant_category_histogram(phonemes):
    return [
        count('voiceless_plosive'),  # カ、タ、パ
        count('voiced_plosive'),     # ガ、ダ、バ
        count('voiceless_fric'),     # サ、シ、ハ
        count('voiced_fric'),        # ザ、ジ
        count('nasal'),              # ン、ム
        count('approximant')         # ラ、ワ、ヤ
    ]
```

**音響的解釈:**
- **無声破裂音** (p, t, k): 鋭いアタック、高周波ノイズ
  - 例: "カッ"、"パン"
- **有声破裂音** (b, d, g): 低周波エネルギー、重み
  - 例: "ガッ"、"ドン"、"ゴロ"
- **無声摩擦音** (s, sh, f, h): 継続的な高周波ノイズ
  - 例: "サラサラ"、"シャー"
- **有声摩擦音** (z, j): 継続的な中〜低周波ノイズ
  - 例: "ザラザラ"、"ジー"
- **鼻音** (m, n, N): 共鳴、こもり、低周波
  - 例: "ン"（撥音）
- **接近音** (r, w, y): 流音、滑らか
  - 例: "リン"、"ワン"

#### グループE: 子音比率のサマリ（3次元）

```python
def _extract_consonant_ratio_summary(phonemes):
    C_count = total_consonants(phonemes)

    # 阻害音（破裂音+摩擦音）の割合
    obstruent_ratio = (plosive + fricative) / C_count

    # 有声子音の割合
    voiced_cons_ratio = (voiced_plosive + voiced_fric) / C_count

    # 鼻音の割合
    nasal_ratio = nasal / C_count

    return [obstruent_ratio, voiced_cons_ratio, nasal_ratio]
```

**音響的解釈:**
- `obstruent_ratio`（阻害音比率）: ノイジーさ、粗さ
  - 高い → 明瞭、くっきり（カキクケコ、サシスセソ）
- `voiced_cons_ratio`（有声比率）: 低音成分の強さ
  - 高い → 重厚、濁り（ガギグゲゴ、ザジズゼゾ）
- `nasal_ratio`（鼻音比率）: 共鳴、こもり
  - 高い → 丸み、柔らかさ

#### グループF: 位置情報（14次元）

```python
def _extract_position_features(moras):
    # 語頭の子音カテゴリ（6次元ワンホット）
    first_consonant_category = detect_first_consonant(moras[0])
    first_onehot = one_hot_encode(first_consonant_category, 6)

    # 語末の子音カテゴリ（6次元ワンホット）
    last_consonant_category = detect_last_consonant(moras[-1])
    last_onehot = one_hot_encode(last_consonant_category, 6)

    # 語頭・語末が母音で始まる/終わるか
    starts_with_vowel = 1.0 if moras[0][0] in vowels else 0.0
    ends_with_vowel = 1.0 if moras[-1][-1] in vowels else 0.0

    return first_onehot + last_onehot + [starts_with_vowel, ends_with_vowel]
```

**音響的解釈:**
- **語頭の子音**: 音の立ち上がり（アタック）の性質
  - "カーン" vs "ガーン" → アタックの鋭さが異なる
- **語末の子音**: 音の終わり方（リリース、減衰）
  - "カン" vs "カー" → 終わり方が異なる
- **母音始まり/終わり**: 柔らかさ vs 明瞭さ

### 3.4 実装例

```python
# "ガンガン" の特徴量抽出例
phonemes = ['g', 'a', 'N', 'g', 'a', 'N']
moras = [('g', 'a'), ('N',), ('g', 'a'), ('N',)]

features = [
    # A: 全体構造（6次元）
    4.0,    # M = 4モーラ
    2.0,    # C_count = 2子音（g × 2）
    2.0,    # V_count = 2母音（a × 2）
    2.0,    # word_repeat_count = 2回（"ガン" が2回）
    0.0,    # mora_repeat_chunk_count = 0
    0.0,    # mora_repeat_ratio = 0

    # B: 長さ（4次元）
    0.0,    # Q_count = 0（促音なし）
    0.0,    # H_mora_count = 0（長音なし）
    0.0,    # H_ratio = 0
    0.0,    # ending_is_long = 0

    # C: 母音ヒストグラム（5次元）
    2.0,    # v_a_count = 2
    0.0,    # v_i_count = 0
    0.0,    # v_u_count = 0
    0.0,    # v_e_count = 0
    0.0,    # v_o_count = 0

    # D: 子音カテゴリ（6次元）
    0.0,    # voiceless_plosive = 0
    2.0,    # voiced_plosive = 2（g × 2）
    0.0,    # voiceless_fric = 0
    0.0,    # voiced_fric = 0
    2.0,    # nasal = 2（N × 2）
    0.0,    # approximant = 0

    # E: 子音比率（3次元）
    0.5,    # obstruent_ratio = 2/4（gが破裂音）
    0.5,    # voiced_cons_ratio = 2/4（gが有声）
    0.5,    # nasal_ratio = 2/4（Nが鼻音）

    # F: 位置情報（14次元）
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0,  # 語頭 = voiced_plosive
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # 語末 = nasal
    0.0,    # starts_with_vowel = 0
    0.0,    # ends_with_vowel = 0
]
```

### 3.5 音響的解釈の対応表

| オノマトペの性質 | 対応する特徴量 | DSPパラメータへの影響 |
|--------------|-------------|-------------------|
| 高音・鋭い | `v_i`多い、`voiceless_plosive`多い | `eq_high`, `eq_presence` 増加 |
| 低音・重い | `v_o`/`v_u`多い、`voiced_plosive`多い | `eq_sub`, `eq_low` 増加 |
| 繰り返し | `word_repeat_count`高い | `compression`, `transient_attack` |
| 長音・持続 | `H_mora_count`高い | `transient_sustain`, `time_stretch` 増加 |
| 促音・鋭い | `Q_count`高い | `transient_attack` 増加 |
| 濁音・重厚 | `voiced_cons_ratio`高い | `eq_sub`, `eq_low` 増加、`gain` 増加 |
| 摩擦音・ノイジー | `voiceless_fric`/`voiced_fric`多い | `eq_high`, `eq_presence` |

---

## 差分モデルの詳細

### 4.1 なぜ差分モデルか？

従来のアプローチ（オノマトペ → DSP）ではなく、**差分アプローチ**を採用：

```python
# ❌ 従来: 絶対的なマッピング
"ゴロゴロ" → DSP parameters

# ✅ 本システム: 相対的なマッピング
("チリン" - "ゴロゴロ") → ΔDSP parameters
```

**利点:**
1. **相対的な音質変化**を直接モデル化
2. **入力音声の特性**（source）を考慮できる
3. より**直感的な操作**（"今チリンなら、ゴロゴロにするには..."）

### 4.2 差分計算

```python
# ステップ1: 各オノマトペから特徴量抽出
φ_source = extract_features("チリン")  # 38次元
φ_target = extract_features("ゴロゴロ")  # 38次元

# ステップ2: 差分計算
Δφ = φ_target - φ_source  # 38次元

# 例:
# φ_source[0] = 3.0 (モーラ数: "チリン" = 3モーラ)
# φ_target[0] = 4.0 (モーラ数: "ゴロゴロ" = 4モーラ)
# Δφ[0] = 4.0 - 3.0 = 1.0 (モーラが1つ増加)
```

**差分の意味:**
- `Δφ[0] > 0`: ターゲットの方がモーラ数が多い → 音が長くなる
- `Δφ[voiced_plosive] > 0`: 有声破裂音が増える → 低音が増える
- `Δφ[v_i] < 0`: 'i'母音が減る → 高音成分が減る

### 4.3 MLPモデル構造

```python
class Onoma2DSPMLP(nn.Module):
    def __init__(self, d_in=38, d_out=10, hidden_dim=32, use_tanh=True):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),   # 38 → 32
            nn.ReLU(),
            nn.Linear(hidden_dim, d_out),  # 32 → 10
            nn.Tanh()                       # 出力を[-1, +1]に制限
        )
```

**アーキテクチャの詳細:**

```
入力層: 38次元（特徴量差分 Δφ）
  ↓
全結合層: 38 → 32
  ↓
ReLU活性化
  ↓
全結合層: 32 → 10
  ↓
Tanh活性化（出力を-1〜+1に制限）
  ↓
出力層: 10次元（正規化されたΔDSP）
```

**パラメータ数:**
- 第1層: 38 × 32 + 32 = 1,248
- 第2層: 32 × 10 + 10 = 330
- **合計: 1,578パラメータ**

**軽量な理由:**
- 複雑な非線形変換は不要（音韻→音響は比較的直接的）
- 過学習を防ぐため小規模に
- リアルタイム処理のため高速に

### 4.4 出力（ΔDSP）の10次元

```python
ΔDSP = [
    Δgain,           # [0] ゲイン変化
    Δcompression,    # [1] 圧縮変化
    Δeq_sub,         # [2] 80Hz EQ変化
    Δeq_low,         # [3] 250Hz EQ変化
    Δeq_mid,         # [4] 1kHz EQ変化
    Δeq_high,        # [5] 4kHz EQ変化
    Δeq_presence,    # [6] 10kHz EQ変化
    Δtransient_attack,  # [7] アタック変化
    Δtransient_sustain, # [8] サスティン変化
    Δtime_stretch    # [9] 時間伸縮変化
]
```

**各次元の範囲と意味:**

| 次元 | パラメータ | 範囲（正規化） | 実際の範囲 | 音響効果 |
|-----|-----------|------------|-----------|---------|
| 0 | gain | [-1, +1] | [-24dB, +24dB] | 音量変化 |
| 1 | compression | [-1, +1] | [-2.0, +2.0] | ダイナミクス圧縮 |
| 2 | eq_sub | [-1, +1] | [-24dB, +24dB] | 超低域（80Hz） |
| 3 | eq_low | [-1, +1] | [-24dB, +24dB] | 低域（250Hz） |
| 4 | eq_mid | [-1, +1] | [-24dB, +24dB] | 中域（1kHz） |
| 5 | eq_high | [-1, +1] | [-24dB, +24dB] | 高域（4kHz） |
| 6 | eq_presence | [-1, +1] | [-24dB, +24dB] | 超高域（10kHz） |
| 7 | transient_attack | [-1, +1] | [-2.0, +2.0] | アタックの鋭さ |
| 8 | transient_sustain | [-1, +1] | [-2.0, +2.0] | サスティンの長さ |
| 9 | time_stretch | [-1, +1] | [0.25x, 2.0x] | 再生速度 |

### 4.5 学習プロセス

```python
# データ準備
X = []  # 特徴量差分のリスト
y = []  # DSPパラメータのリスト

for sample in dataset:
    # 各サンプルは (onomatopoeia, audio_file, dsp_params) のペア
    φ = extract_features(onomatopoeia)
    X.append(φ)
    y.append(normalize_dsp_params(dsp_params))

# 学習
model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=32)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = MSELoss()

for epoch in range(200):
    for Δφ_batch, ΔDSP_batch in dataloader:
        # Forward
        ΔDSP_pred = model(Δφ_batch)
        loss = criterion(ΔDSP_pred, ΔDSP_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**損失関数: MSE (Mean Squared Error)**

```python
loss = MSE(ΔDSP_pred, ΔDSP_true)
     = (1/10) Σ(ΔDSP_pred[i] - ΔDSP_true[i])²
```

**評価指標:**
1. **MSE**: 予測誤差の大きさ
2. **R² Score**: 説明力（1.0に近いほど良い）
3. **符号正解率**: 変化の方向が正しいか（増加/減少）

### 4.6 推論プロセス

```python
# 推論時
def predict(source_onoma, target_onoma):
    # 1. 特徴量抽出
    φ_source = extract_features(source_onoma)
    φ_target = extract_features(target_onoma)

    # 2. 差分計算
    Δφ = φ_target - φ_source

    # 3. 標準化（学習時のスケーラーを使用）
    Δφ_scaled = scaler.transform(Δφ.reshape(1, -1))

    # 4. モデル推論
    with torch.no_grad():
        Δφ_tensor = torch.FloatTensor(Δφ_scaled)
        ΔDSP_norm = model(Δφ_tensor).numpy()[0]
    # ΔDSP_norm の範囲: [-1, +1]

    # 5. Amplification（オプション）
    ΔDSP_norm = np.clip(
        ΔDSP_norm * amplification_factor,
        -1.0, 1.0
    )

    # 6. Attention補正（オプション）
    if lambda_att > 0:
        attention = compute_attention(source_onoma)
        ΔDSP_norm = ΔDSP_norm * (1.0 + lambda_att * attention)
        ΔDSP_norm = np.clip(ΔDSP_norm, -1.0, 1.0)

    # 7. 実際のDSPパラメータにマッピング
    dsp_params = map_to_real_values(ΔDSP_norm)

    return dsp_params
```

---

## DSPパラメータマッピング

### 5.1 マッピング関数

正規化値[-1, +1]を実際のDSPパラメータに変換：

```python
def map_parameters(normalized_params):
    """
    normalized_params: 10次元、範囲[-1, +1]
    戻り値: 実際のパラメータ辞書
    """

    # Gain: -24dB 〜 +24dB
    gain_db = 24.0 * normalized_params[0]

    # Compression: -2.0 〜 +2.0
    compression = 2.0 * normalized_params[1]

    # EQ (5バンド): -24dB 〜 +24dB
    eq_sub_db = 24.0 * normalized_params[2]
    eq_low_db = 24.0 * normalized_params[3]
    eq_mid_db = 24.0 * normalized_params[4]
    eq_high_db = 24.0 * normalized_params[5]
    eq_presence_db = 24.0 * normalized_params[6]

    # Transient: -2.0 〜 +2.0
    transient_attack = 2.0 * normalized_params[7]
    transient_sustain = 2.0 * normalized_params[8]

    # Time Stretch: 0.25倍 〜 2.0倍
    # -1 → 0.25, 0 → 1.0, +1 → 2.0
    time_stretch_ratio = 1.0 + 0.75 * normalized_params[9]

    return {
        'gain_db': gain_db,
        'compression': compression,
        'eq_sub_db': eq_sub_db,
        'eq_low_db': eq_low_db,
        'eq_mid_db': eq_mid_db,
        'eq_high_db': eq_high_db,
        'eq_presence_db': eq_presence_db,
        'transient_attack': transient_attack,
        'transient_sustain': transient_sustain,
        'time_stretch_ratio': time_stretch_ratio
    }
```

### 5.2 各パラメータの音響効果

#### EQパラメータ

```
周波数帯域の配置:

  eq_sub (80Hz)      超低域  「ズーン」「ゴーン」の迫力
    ↓
  eq_low (250Hz)     低域    「ドン」「ゴロ」の重厚感
    ↓
  eq_mid (1kHz)      中域    音の明瞭度、存在感
    ↓
  eq_high (4kHz)     高域    「カン」「キン」の明るさ
    ↓
  eq_presence (10kHz) 超高域  「チリン」「キラキラ」の煌めき
```

**EQの効果例:**

```python
# "チリン" → "ゴロゴロ" の場合
eq_high_db = -18.0      # 高音を18dB減衰（チリンの特徴を抑える）
eq_presence_db = -20.0  # 超高域を20dB減衰
eq_sub_db = +15.0       # 超低域を15dB増幅（ゴロゴロの特徴）
eq_low_db = +18.0       # 低域を18dB増幅
```

#### Transient Shaping

```python
# transient_attack: -2.0 〜 +2.0
# 負の値: アタックを鈍らせる（柔らかく）
# 正の値: アタックを鋭くする（明瞭に）

# transient_sustain: -2.0 〜 +2.0
# 負の値: サスティンを短く（歯切れ良く）
# 正の値: サスティンを長く（余韻を伸ばす）
```

**効果例:**

```python
# "カッ" → "ガッ" の場合
transient_attack = -0.8   # アタックをやや鈍らせる
transient_sustain = +0.3  # サスティンをやや伸ばす

# "ポン" → "バン" の場合
transient_attack = +1.2   # アタックを鋭くする
```

---

## Attention機構

### 6.1 概念

**問題意識:**
- ユーザーが"チリン"と言うとき、**高音域に注目している**
- "ゴロゴロ"と言うとき、**低音域に注目している**
- この「注目」情報を活用してDSP変化を補正

**Attention機構の役割:**
```
ソースオノマトペ → ユーザーの注目領域を推定
                 ↓
        その領域の変化を強調
```

### 6.2 実装

```python
def apply_attention_correction(normalized_params, source_onoma, lambda_att):
    """
    Attention補正を適用

    Args:
        normalized_params: MLPの出力（10次元、-1〜+1）
        source_onoma: ソースオノマトペ（例: "チリン"）
        lambda_att: Attention強度（0.0〜1.0）

    Returns:
        補正後のパラメータ
    """

    # 1. ソースオノマトペのDSPテンプレートを生成
    template = create_dsp_template(source_onoma)
    # template: [gain, comp, eq_sub, eq_low, eq_mid,
    #            eq_high, eq_pres, atk, sus, stretch]

    # 2. 絶対値を取って注目度ベクトルにする
    attention = np.abs(template)
    # 理由: 正負の符号は関係なく、「その次元が重要か」を知りたい

    # 3. 0-1に正規化
    attention = attention / np.max(attention) if np.max(attention) > 0 else attention

    # 4. 補正式を適用
    corrected = normalized_params * (1.0 + lambda_att * attention)

    # 5. クリッピング
    corrected = np.clip(corrected, -1.0, 1.0)

    return corrected
```

### 6.3 create_dsp_template の詳細

```python
def create_dsp_template(onomatopoeia):
    """
    オノマトペからヒューリスティックなDSPテンプレートを生成

    Returns:
        10次元のDSPパラメータ（正規化値 -1〜+1）
    """
    # 音素・モーラに変換
    phonemes = katakana_to_phoneme(onomatopoeia)
    moras = phoneme_to_mora(phonemes)

    # 初期値
    params = [0.0] * 10

    # 濁音カウント（g, d, z, b）
    voiced_count = count_voiced_consonants(phonemes)

    # 高音系子音（k, p, t, s, sh, ch, ts）
    high_consonants = count_high_consonants(phonemes)

    # 促音・長音
    sokuon_count = count_Q(phonemes)
    choon_count = count_H(phonemes)

    # === ルールベースでパラメータを設定 ===

    # 1. Gain: 濁音が多い → 大音量
    if voiced_count >= 2:
        params[0] = 0.3 + 0.2 * min(voiced_count, 4)
    elif high_consonants >= 1:
        params[0] = -0.2 - 0.1 * min(high_consonants, 3)

    # 2. Compression
    if voiced_count >= 2:
        params[1] = 0.3 + 0.1 * min(voiced_count, 4)

    # 3. EQ Sub（超低域）
    if voiced_count >= 2:
        params[2] = 0.4 + 0.2 * min(voiced_count, 4)  # 濁音 → 低音強化
    elif high_consonants >= 1:
        params[2] = -0.3 - 0.1 * min(high_consonants, 3)  # 高音系 → 低音カット

    # 4. EQ Low（低域）
    if voiced_count >= 1:
        params[3] = 0.3 + 0.2 * min(voiced_count, 4)
    elif high_consonants >= 1:
        params[3] = -0.2 - 0.1 * min(high_consonants, 3)

    # 5. EQ Mid（中域）
    params[4] = 0.0  # ニュートラル

    # 6. EQ High（高域）
    if high_consonants >= 1:
        params[5] = 0.4 + 0.2 * min(high_consonants, 4)  # 高音系 → 高音強調
    elif voiced_count >= 2:
        params[5] = -0.2 - 0.1 * min(voiced_count, 3)  # 濁音 → 高音カット

    # 7. EQ Presence（超高域）
    if high_consonants >= 1:
        params[6] = 0.5 + 0.2 * min(high_consonants, 4)
    elif voiced_count >= 2:
        params[6] = -0.2 - 0.1 * min(voiced_count, 3)

    # 8. Transient Attack
    if sokuon_count > 0:
        params[7] = 0.7  # 促音 → 鋭いアタック
    elif choon_count > 1:
        params[7] = -0.4  # 長音 → 柔らかいアタック

    # 9. Transient Sustain
    if choon_count > 1:
        params[8] = 0.6  # 長音 → 長いサスティン
    elif sokuon_count > 0:
        params[8] = -0.3  # 促音 → 短い

    # 10. Time Stretch
    if choon_count > 2:
        params[9] = 0.3  # 長音多い → やや伸ばす
    elif sokuon_count > 1:
        params[9] = -0.2  # 促音多い → やや短く

    return params
```

### 6.4 Attentionの効果

**例: "チリン" → "ゴロゴロ"**

```python
# 1. ソーステンプレート生成
template_chirin = create_dsp_template("チリン")
# = [-0.3, 0, -0.4, -0.3, 0, 0.6, 0.7, 0, 0, 0]
#     ↑                     ↑    ↑
#   gain低め            高音系が強い

# 2. 注目度ベクトル
attention = np.abs(template_chirin)
# = [0.3, 0, 0.4, 0.3, 0, 0.6, 0.7, 0, 0, 0]

# 3. 正規化
attention = attention / 0.7  # max値で割る
# = [0.429, 0, 0.571, 0.429, 0, 0.857, 1.000, 0, 0, 0]
#                              ↑      ↑
#                        高音域に高い注目度

# 4. MLPの出力（例）
dsp_pred = [-0.3, 0.1, 0.5, 0.5, 0.0, -0.65, -0.85, -0.5, 0.2, -0.05]

# 5. Attention補正（lambda_att = 0.7）
corrected = dsp_pred * (1.0 + 0.7 * attention)
#
# eq_high次元（[5]）:
#   -0.65 * (1.0 + 0.7 * 0.857)
#   = -0.65 * 1.600
#   = -1.04 → クリップして -1.0
#
# eq_presence次元（[6]）:
#   -0.85 * (1.0 + 0.7 * 1.000)
#   = -0.85 * 1.700
#   = -1.445 → クリップして -1.0

# 結果: 高音カットがより強調される！
```

**効果まとめ:**
- ソースオノマトペの特徴的な次元の変化が**増幅**される
- ユーザーの「聴いているポイント」に焦点を当てた変換
- より**意図に沿った**、**自然な**変換が可能

---

## 学習データとプロセス

### 7.1 データセット: RWCP-SSD-Onomatopoeia

**構成:**
```
training_data_jp_utf8bom.csv
├─ audio_path: 音声ファイルパス
├─ onomatopoeia: オノマトペ（カタカナ）
├─ confidence: アノテーション信頼度（1-5）
└─ avg_acceptability: 受容度（1-5）
```

**フィルタリング:**
```python
df_filtered = df[
    (df['confidence'] >= 4) &          # 高信頼度のみ
    (df['avg_acceptability'] >= 4.0)   # 高受容度のみ
]
```

### 7.2 データ準備

```python
def create_rwcp_dataset():
    """
    RWCP-SSDデータからMLPモデル用のデータセットを作成
    """

    # 1. CSVを読み込み
    df = pd.read_csv('training_data_jp_utf8bom.csv')

    # 2. フィルタリング
    df = df[(df['confidence'] >= 4) & (df['avg_acceptability'] >= 4.0)]

    # 3. 各サンプルに対して
    for idx, row in df.iterrows():
        onomatopoeia = row['onomatopoeia']
        audio_path = row['audio_path']

        # オノマトペ特徴量抽出
        φ = extract_features(onomatopoeia)

        # ヒューリスティックなDSPパラメータ生成
        dsp_template = create_dsp_template(onomatopoeia)

        # データに追加
        dataset.append({
            'onomatopoeia': onomatopoeia,
            'audio_path': audio_path,
            'features': φ,
            'dsp_params': dsp_template
        })

    return dataset
```

**重要:** 学習データの DSP パラメータは`create_dsp_template()`で**ヒューリスティックに生成**されます。実際の音声信号から抽出したものではありません。

### 7.3 学習設定

```python
# モデル
model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=32, use_tanh=True)

# 最適化
optimizer = Adam(model.parameters(), lr=0.001)
criterion = MSELoss()

# データローダー
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 学習
epochs = 200
for epoch in range(epochs):
    # 訓練
    for Δφ_batch, ΔDSP_batch in train_loader:
        output = model(Δφ_batch)
        loss = criterion(output, ΔDSP_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 検証
    val_loss = evaluate(model, val_loader)
```

---

## 実装の詳細

### 8.1 ファイル構成

```
Tsuji_MLP/
├── src/
│   ├── preprocessing/
│   │   ├── katakana_to_phoneme.py       # カタカナ→音素
│   │   ├── phoneme_to_mora.py           # 音素→モーラ
│   │   └── feature_extractor.py         # 38次元特徴量抽出
│   ├── models/
│   │   └── mlp_model.py                 # MLPモデル定義
│   ├── dsp/
│   │   └── dsp_engine.py                # DSP処理エンジン
│   ├── data/
│   │   └── data_loader.py               # データローダー
│   ├── utils/
│   │   └── create_rwcp_dataset.py       # データセット生成
│   ├── onoma2dsp.py                      # メインシステム
│   ├── cli.py                            # CLIインターフェース
│   └── train_with_rwcp.py                # 学習スクリプト
├── models/
│   ├── rwcp_model.pth                    # 学習済みモデル
│   └── rwcp_scaler.pkl                   # StandardScaler
├── data/
│   └── rwcp_dataset.csv                  # 学習用データセット
├── history/
│   └── edit_history.json                 # 編集履歴
└── output/
    └── *.wav                             # 処理済み音声
```

### 8.2 処理フロー（コード付き）

```python
# メインシステム (onoma2dsp.py)
class Onoma2DSP:
    def process(self, source_onoma, target_onoma, input_audio, output_audio):
        # 1. オノマトペ前処理
        source_phonemes = self.katakana_converter.convert(source_onoma)
        target_phonemes = self.katakana_converter.convert(target_onoma)

        source_moras = self.mora_converter.convert(source_phonemes)
        target_moras = self.mora_converter.convert(target_phonemes)

        # 2. 特徴量抽出
        φ_source = self.feature_extractor.extract_features(
            source_phonemes, source_moras
        )
        φ_target = self.feature_extractor.extract_features(
            target_phonemes, target_moras
        )

        # 3. 差分計算
        Δφ = φ_target - φ_source

        # 4. 標準化
        Δφ_scaled = self.scaler.transform(Δφ.reshape(1, -1))

        # 5. モデル推論
        Δφ_tensor = torch.FloatTensor(Δφ_scaled)
        with torch.no_grad():
            ΔDSP_norm = self.model(Δφ_tensor).numpy()[0]

        # 6. Amplification
        ΔDSP_norm = np.clip(
            ΔDSP_norm * self.amplification_factor,
            -1.0, 1.0
        )

        # 7. Attention補正
        if self.lambda_att > 0:
            template_source = create_dsp_template(source_onoma)
            attention = np.abs(template_source)
            attention = attention / np.max(attention) if np.max(attention) > 0 else attention

            ΔDSP_norm = ΔDSP_norm * (1.0 + self.lambda_att * attention)
            ΔDSP_norm = np.clip(ΔDSP_norm, -1.0, 1.0)

        # 8. パラメータマッピング
        dsp_params = self.mapper.map_parameters(ΔDSP_norm)

        # 9. DSP処理
        self.dsp_engine.process_audio_file(
            input_audio, output_audio, dsp_params
        )

        return {
            'source_onomatopoeia': source_onoma,
            'target_onomatopoeia': target_onoma,
            'feature_diff_magnitude': float(np.linalg.norm(Δφ)),
            'mapped_params': dsp_params,
            'output_audio': output_audio
        }
```

### 8.3 パフォーマンス

**処理速度:**
- 0.5秒の音声: 約0.7秒で処理完了
- 特徴量抽出: <0.01秒
- モデル推論: <0.01秒
- DSP処理: 音声長に比例

**メモリ使用量:**
- モデルサイズ: 約6KB（1,578パラメータ）
- 実行時メモリ: <100MB

**推奨環境:**
- CPU: 2コア以上
- RAM: 2GB以上
- Python: 3.8以上

---

## まとめ

### システムの革新性

1. **差分ベースのアプローチ**: 絶対的ではなく相対的な音質変化をモデル化
2. **軽量な設計**: 1,578パラメータのみで効果的な変換
3. **Attention機構**: ユーザーの聴覚的注目を推定して補正
4. **解釈可能性**: オノマトペの音韻的特徴と音響パラメータの対応が明確

### 技術的特徴

- **入力**: 38次元の音韻的特徴量（6グループ）
- **モデル**: 軽量MLP（38→32→10）
- **出力**: 10次元のDSPパラメータ変化
- **拡張**: Attention機構による適応的補正

### 使用シーン

- 効果音の音質変換（0.5〜3秒が最適）
- 音楽ループの加工（2〜10秒）
- 直感的な音声編集（オノマトペで指示）

---

**参考文献:**
- RWCP-SSD-Onomatopoeia Dataset
- PyTorch Documentation
- Digital Signal Processing Theory

**バージョン履歴:**
- v1.0 (2025-12-03): 初版作成
