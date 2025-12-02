# 差分オノマトペシステム - 完全技術ドキュメント

## 目次

1. [システム概要](#1-システム概要)
2. [背景と動機](#2-背景と動機)
3. [システムアーキテクチャ](#3-システムアーキテクチャ)
4. [データフロー](#4-データフロー)
5. [コンポーネント詳細](#5-コンポーネント詳細)
6. [特徴量抽出の詳細](#6-特徴量抽出の詳細)
7. [DSPパラメータの詳細](#7-dspパラメータの詳細)
8. [モデル構造](#8-モデル構造)
9. [学習プロセス](#9-学習プロセス)
10. [ファイル構成](#10-ファイル構成)
11. [使用方法](#11-使用方法)
12. [実装の詳細](#12-実装の詳細)
13. [パラメータ調整](#13-パラメータ調整)
14. [制約事項と課題](#14-制約事項と課題)
15. [今後の拡張性](#15-今後の拡張性)

---

## 1. システム概要

### 1.1 概要

**差分オノマトペシステム**は、2つの日本語オノマトペ（擬音語・擬態語）の特徴量差分を利用して、音声に対するDSP（Digital Signal Processing）パラメータを自動生成し、音声を変換するシステムです。

### 1.2 キーコンセプト

従来の「単一オノマトペ → DSPパラメータ」方式ではなく、**「現在の音を表すオノマトペ」と「目標とする音を表すオノマトペ」の2つを入力**として受け取り、その差分から音声変換を行います。

```
入力: source_onomatopoeia（現在の音） + target_onomatopoeia（目標の音）
処理: Δ = target特徴量 - source特徴量 → モデル → DSPパラメータ
出力: 変換された音声
```

### 1.3 システムの特徴

- **直感的な操作**: 「チリンをゴロゴロのような音にしたい」という自然な指示が可能
- **差分ベース**: 音の変化量を直接制御
- **MLPモデル**: 38次元の特徴量差分から10次元のDSPパラメータを予測
- **実音声処理**: EQ、コンプレッション、トランジェント、タイムストレッチなどを適用
- **RWCP-SSD-Onomatopoeiaデータセット**: 8,542サンプル、1,811種類のオノマトペで学習

---

## 2. 背景と動機

### 2.1 従来システムの課題

旧システムでは、**単一のオノマトペ**を入力として受け取り、そのオノマトペが表す音響特性に基づいてDSPパラメータを生成していました。

**問題点:**
1. **文脈がない**: 元の音がどのような特徴を持っているか考慮されない
2. **絶対値ベース**: 「ゴロゴロ」と入力すると、元の音に関係なく同じパラメータが生成される
3. **変化量の制御が困難**: 「少しだけ重くする」「大幅に明るくする」といった調整が難しい

### 2.2 差分ベースの利点

**新システムの利点:**
1. **相対的な変換**: 元の音の特徴を考慮した変換が可能
2. **直感的**: 「Aという音をBのような音にする」という自然な表現
3. **柔軟性**: 同じターゲットでもソースが異なれば異なる変換が適用される
4. **制御性**: 特徴量の差分量が変換の強度を決定

### 2.3 実例

```
例1: ベル音（元々明るい音）
  source: チリン → target: ゴロゴロ
  → 大幅に低音を強調し、明るさを削減

例2: 木材音（元々暗い音）
  source: カッ → target: ゴロゴロ
  → 既にやや暗いため、低音強調は控えめ
```

---

## 3. システムアーキテクチャ

### 3.1 全体構成

```
┌─────────────────────────────────────────────────────────────┐
│                    差分オノマトペシステム                      │
└─────────────────────────────────────────────────────────────┘

入力:
  - source_onomatopoeia (str): 現在の音を表すオノマトペ
  - target_onomatopoeia (str): 目標の音を表すオノマトペ
  - input_audio_path (str): 入力音声ファイル

                            ↓

┌─────────────────────────────────────────────────────────────┐
│ Step 1: 音韻変換 (KatakanaToPhoneme)                         │
│   - カタカナ → 音素列に変換                                   │
│   - 例: "チリン" → ['ch', 'i', 'r', 'i', 'N']                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: モーラ分割 (PhonemeToMora)                           │
│   - 音素列 → モーラ（拍）に分割                               │
│   - 例: ['ch','i','r','i','N'] → ['chi','ri','N']          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 特徴量抽出 (OnomatopoeiaFeatureExtractor)            │
│   - 音素とモーラから38次元の特徴ベクトル抽出                   │
│   - Source用とTarget用で2回実行                              │
│   - 特徴量: 母音/子音カウント、濁音、促音、長音、繰り返し等    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 差分計算                                             │
│   feature_diff = target_features - source_features          │
│   - 38次元のベクトル差分                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: スケーリング (StandardScaler)                        │
│   - 学習時に計算した平均・分散で正規化                         │
│   - feature_diff_scaled = scaler.transform(feature_diff)    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: モデル推論 (Onoma2DSPMLP)                            │
│   - MLP: 38 → 32 → 10                                       │
│   - 入力: 38次元の差分特徴量                                  │
│   - 出力: 10次元の正規化DSPパラメータ (-1 〜 +1)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 7: パラメータマッピング (DSPParameterMapping)            │
│   - 正規化値 → 実際のdB値・倍率に変換                         │
│   - amplification_factor=5.0 で増幅                          │
│   - 例: -1〜+1 → -24dB〜+24dB (gain)                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 8: DSP処理 (DSPEngine)                                 │
│   - EQ: 5バンド (sub/low/mid/high/presence)                │
│   - Compression: 音量圧縮                                    │
│   - Transient Shaper: アタック・サステイン調整               │
│   - Time Stretch: 再生速度変更                               │
│   - Gain: 最終的な音量調整                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓

出力:
  - processed_audio_file: 変換された音声ファイル
  - result (dict): 処理結果の詳細情報
```

### 3.2 レイヤー構成

1. **入力層**: カタカナ文字列（オノマトペ2つ）
2. **前処理層**: 音韻・モーラ変換
3. **特徴抽出層**: 38次元ベクトル生成 × 2
4. **差分計算層**: 特徴量の減算
5. **正規化層**: StandardScaler
6. **推論層**: MLP (38→32→10)
7. **増幅層**: amplification_factor適用
8. **マッピング層**: 正規化値 → 実値変換
9. **DSP処理層**: 実音声処理
10. **出力層**: 処理済み音声ファイル

---

## 4. データフロー

### 4.1 詳細なデータフロー図

```
Source: "チリン"     Target: "ゴロゴロ"
      ↓                   ↓
  音韻変換             音韻変換
      ↓                   ↓
['ch','i','r','i','N']  ['g','o','r','o','g','o','r','o']
      ↓                   ↓
  モーラ分割           モーラ分割
      ↓                   ↓
['chi','ri','N']      ['go','ro','go','ro']
      ↓                   ↓
  特徴抽出             特徴抽出
      ↓                   ↓
[38次元ベクトルA]     [38次元ベクトルB]
      ↓                   ↓
      └───────┬───────┘
              ↓
        差分計算 (B - A)
              ↓
      [38次元差分ベクトル]
      magnitude: 9.784
              ↓
        スケーリング
              ↓
      [38次元正規化差分]
              ↓
        MLPモデル推論
        (38 → 32 → 10)
              ↓
      [10次元正規化パラメータ]
      範囲: -1.0 〜 +1.0
              ↓
        増幅 (×5.0)
              ↓
      [10次元増幅パラメータ]
      範囲: -1.0 〜 +1.0 (clipped)
              ↓
        パラメータマッピング
              ↓
┌───────────────────────────┐
│ 実DSPパラメータ              │
│ - gain_db: +21.37 dB       │
│ - compression: 0.35        │
│ - eq_sub_db: +24.00 dB     │
│ - eq_low_db: +24.00 dB     │
│ - eq_mid_db: +0.02 dB      │
│ - eq_high_db: -15.62 dB    │
│ - eq_presence_db: -20.39dB │
│ - transient_attack: -1.58  │
│ - transient_sustain: 0.56  │
│ - time_stretch_ratio: 0.89x│
└───────────────────────────┘
              ↓
        DSPエンジン
        (input_audio)
              ↓
      処理済み音声ファイル
```

### 4.2 各ステップでのデータ型

| ステップ | データ型 | 次元/形式 | 例 |
|---------|---------|----------|-----|
| 入力オノマトペ | str | 可変長文字列 | "チリン" |
| 音素列 | List[str] | 可変長リスト | ['ch','i','r','i','N'] |
| モーラ | List[str] | 可変長リスト | ['chi','ri','N'] |
| 特徴量 | np.ndarray | (38,) | [0.2, 0.4, ...] |
| 差分特徴量 | np.ndarray | (38,) | [0.1, -0.3, ...] |
| 正規化差分 | np.ndarray | (38,) | [0.5, -1.2, ...] |
| モデル出力 | np.ndarray | (10,) | [-0.3, 0.7, ...] |
| 増幅出力 | np.ndarray | (10,) | [-1.0, 1.0, ...] |
| DSPパラメータ | dict | 10キー | {'gain_db': 21.37, ...} |
| 音声データ | np.ndarray | (samples,) | 44.1kHz波形 |

---

## 5. コンポーネント詳細

### 5.1 KatakanaToPhoneme（音韻変換器）

**役割**: カタカナ文字列を音素列に変換

**実装**: `src/preprocessing/katakana_to_phoneme.py`

**変換ルール**:
```python
# 基本的な変換例
'カ' → ['k', 'a']
'ガ' → ['g', 'a']  # 濁音
'キャ' → ['ky', 'a']  # 拗音
'ッ' → ['Q']  # 促音
'ン' → ['N']  # 撥音
'ー' → ['H']  # 長音
```

**特殊な処理**:
- 拗音（きゃ、しゅ、ちょ等）の処理
- 促音（っ）の処理
- 長音（ー）の処理
- 撥音（ん）の処理

**コード例**:
```python
converter = KatakanaToPhoneme()
phonemes = converter.convert("チリン")
# → ['ch', 'i', 'r', 'i', 'N']
```

### 5.2 PhonemeToMora（モーラ変換器）

**役割**: 音素列をモーラ（拍）に分割

**実装**: `src/preprocessing/phoneme_to_mora.py`

**モーラの定義**:
- 日本語の最小リズム単位
- 1モーラ = 基本的に1拍
- 「キャ」= 1モーラ、「カ」= 1モーラ

**変換ルール**:
```python
# 基本的なモーラ分割
['k', 'a', 'r', 'a'] → ['ka', 'ra']
['k', 'y', 'a'] → ['kya']  # 拗音は1モーラ
['k', 'a', 'Q'] → ['ka', 'Q']  # 促音は単独モーラ
['k', 'a', 'N'] → ['ka', 'N']  # 撥音は単独モーラ
```

**コード例**:
```python
converter = PhonemeToMora()
moras = converter.convert(['ch', 'i', 'r', 'i', 'N'])
# → ['chi', 'ri', 'N']
```

### 5.3 OnomatopoeiaFeatureExtractor（特徴量抽出器）

**役割**: 音素列とモーラから38次元の特徴ベクトルを生成

**実装**: `src/preprocessing/feature_extractor.py`

**特徴量の構成** (38次元):

#### 基本統計量 (8次元)
1. `num_phonemes`: 音素数
2. `num_moras`: モーラ数
3. `num_vowels`: 母音の数
4. `num_consonants`: 子音の数
5. `vowel_ratio`: 母音比率
6. `consonant_ratio`: 子音比率
7. `mora_per_phoneme`: モーラ/音素比
8. `avg_mora_length`: 平均モーラ長

#### 母音特徴量 (5次元)
9-13. `vowel_a/i/u/e/o_count`: 各母音('a','i','u','e','o')の出現回数

#### 子音特徴量 (20次元)
14-33. 子音グループ別カウント:
- 無声子音: k, s, t, h, p
- 有声子音: g, z, d, b
- 鼻音: n, m
- 流音: r
- 半母音: y, w
- 破擦音: ch, j
- 摩擦音: sh

#### 特殊音素 (3次元)
34. `sokuon_count`: 促音（っ）の数
35. `hatsuon_count`: 撥音（ん）の数
36. `chouon_count`: 長音（ー）の数

#### 高次特徴量 (2次元)
37. `has_dakuon`: 濁音の有無（0 or 1）
38. `has_repetition`: 繰り返しパターンの有無（0 or 1）

**コード例**:
```python
extractor = OnomatopoeiaFeatureExtractor()
phonemes = ['g', 'o', 'r', 'o', 'g', 'o', 'r', 'o']
moras = ['go', 'ro', 'go', 'ro']
features = extractor.extract_features(phonemes, moras)
# → np.ndarray, shape=(38,)
```

**特徴量の例**:

```
"ゴロゴロ" の特徴量:
  num_phonemes: 8
  num_moras: 4
  vowel_o_count: 4  # 'o'が4回
  consonant_g_count: 2  # 'g'が2回
  consonant_r_count: 2  # 'r'が2回
  has_dakuon: 1  # 濁音あり
  has_repetition: 1  # 繰り返しあり
```

### 5.4 Onoma2DSPMLP（MLPモデル）

**役割**: 38次元の特徴量差分から10次元のDSPパラメータを予測

**実装**: `src/models/mlp_model.py`

**アーキテクチャ**:
```
Input Layer:    38次元
    ↓
Hidden Layer:   32次元 (ReLU)
    ↓
Output Layer:   10次元 (Tanh)
```

**層の詳細**:
```python
nn.Sequential(
    nn.Linear(38, 32),  # 全結合層1
    nn.ReLU(),          # 活性化関数
    nn.Linear(32, 10),  # 全結合層2
    nn.Tanh()           # 出力を-1〜+1に制限
)
```

**パラメータ数**:
- Layer1: 38×32 + 32 = 1,248
- Layer2: 32×10 + 10 = 330
- **合計**: 1,578パラメータ

**学習済みモデル**:
- ファイル: `models/rwcp_model.pth`
- 訓練データ: RWCP-SSD-Onomatopoeia (8,542サンプル)
- 性能: R² = 0.8036, MSE = 0.000900

**コード例**:
```python
model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=32, use_tanh=True)
model.load_state_dict(torch.load('models/rwcp_model.pth'))
model.eval()

with torch.no_grad():
    feature_diff = torch.FloatTensor(diff_vector).unsqueeze(0)
    params = model(feature_diff).numpy()[0]
    # → shape=(10,), 範囲[-1, +1]
```

### 5.5 DSPParameterMapping（パラメータマッピング）

**役割**: 正規化されたパラメータ(-1〜+1)を実際のdB値・倍率に変換

**実装**: `src/models/mlp_model.py`

**マッピングルール**:

| 正規化パラメータ | マッピング式 | 範囲 |
|----------------|-------------|------|
| `normalized[0]` | `24.0 × val` | gain_db: -24〜+24 dB |
| `normalized[1]` | `2.0 × val` | compression: -2〜+2 |
| `normalized[2]` | `24.0 × val` | eq_sub_db: -24〜+24 dB |
| `normalized[3]` | `24.0 × val` | eq_low_db: -24〜+24 dB |
| `normalized[4]` | `24.0 × val` | eq_mid_db: -24〜+24 dB |
| `normalized[5]` | `24.0 × val` | eq_high_db: -24〜+24 dB |
| `normalized[6]` | `24.0 × val` | eq_presence_db: -24〜+24 dB |
| `normalized[7]` | `2.0 × val` | transient_attack: -2〜+2 |
| `normalized[8]` | `2.0 × val` | transient_sustain: -2〜+2 |
| `normalized[9]` | `1.0 + 0.75 × val` | time_stretch: 0.25〜2.0x |

**増幅処理**:

モデル出力を増幅係数でスケーリング:
```python
amplification_factor = 5.0  # デフォルト
normalized_params = np.clip(
    model_output * amplification_factor,
    -1.0,
    +1.0
)
```

**コード例**:
```python
mapper = DSPParameterMapping()
normalized = np.array([0.5, 0.3, -0.8, ...])  # 10次元
real_params = mapper.map_parameters(normalized)
# → {
#     'gain_db': 12.0,
#     'compression': 0.6,
#     'eq_sub_db': -19.2,
#     ...
# }
```

### 5.6 DSPEngine（DSP処理エンジン）

**役割**: 実際の音声にDSPエフェクトを適用

**実装**: `src/dsp/dsp_engine.py`

**処理順序**:
1. EQ（イコライザー）
2. Compression（コンプレッション）
3. Transient Shaper（トランジェント整形）
4. Time Stretch（再生速度変更）
5. Gain（最終音量調整）

**各エフェクトの詳細**:

#### 1. EQ（5バンド・イコライザー）
```python
周波数帯域:
- sub: 60 Hz（超低音）
- low: 250 Hz（低音）
- mid: 1000 Hz（中音）
- high: 4000 Hz（高音）
- presence: 8000 Hz（超高音）

実装: scipy.signal.butter (2次バターワースフィルタ)
```

#### 2. Compression（音量圧縮）
```python
目的: 音量のダイナミックレンジを圧縮
パラメータ: compression (-2〜+2)
  - 正の値: より強い圧縮
  - 負の値: より弱い圧縮
実装: しきい値ベースの音量調整
```

#### 3. Transient Shaper（トランジェント整形）
```python
目的: アタック（立ち上がり）とサステイン（持続）を調整

transient_attack (-2〜+2):
  - 正の値: アタックを強調
  - 負の値: アタックを弱める

transient_sustain (-2〜+2):
  - 正の値: サステインを延長
  - 負の値: サステインを短縮

実装: エンベロープ検出 + ゲイン調整
```

#### 4. Time Stretch（再生速度変更）
```python
time_stretch_ratio (0.25〜2.0x):
  - < 1.0: 遅く、低く
  - = 1.0: 変更なし
  - > 1.0: 速く、高く

実装: librosa.effects.time_stretch
```

#### 5. Gain（音量調整）
```python
gain_db (-24〜+24 dB):
  - 正の値: 音量を上げる
  - 負の値: 音量を下げる

変換式:
  linear_gain = 10^(gain_db / 20)
  output = input × linear_gain
```

**コード例**:
```python
dsp = DSPEngine(sample_rate=44100)
params = {
    'gain_db': 12.0,
    'compression': 0.5,
    'eq_sub_db': 6.0,
    'eq_low_db': 3.0,
    # ... 他のパラメータ
}
dsp.process_audio_file(
    'input.wav',
    'output.wav',
    params
)
```

---

## 6. 特徴量抽出の詳細

### 6.1 特徴量設計の哲学

日本語のオノマトペは、音韻構造と意味（音響イメージ）が密接に関連しています：

- **濁音** (が、ざ、だ、ば): 重い、暗い、力強い
- **清音** (か、さ、た、は): 軽い、明るい、繊細
- **促音** (っ): 鋭い、短い
- **長音** (ー): 伸びる、持続する
- **繰り返し** (ゴロゴロ): 連続性、リズム

これらの音韻特徴を数値化することで、音響特性を予測可能にします。

### 6.2 各特徴量の音響的意味

#### 基本統計量

**num_phonemes / num_moras**:
- 多い → 複雑な音
- 少ない → シンプルな音

**vowel_ratio / consonant_ratio**:
- 母音多い → 響きが豊か
- 子音多い → リズミカル

#### 母音特徴量

日本語の5母音は周波数特性と対応:
- **'a'** (あ): 低い、開いた音
- **'i'** (い): 高い、狭い音
- **'u'** (う): 中程度、丸い音
- **'e'** (え): やや高い音
- **'o'** (お): やや低い、丸い音

#### 子音特徴量

子音タイプと音響特性:
- **無声子音** (k, s, t, h, p): 鋭い、軽い
- **有声子音** (g, z, d, b): 重い、暗い
- **鼻音** (n, m): こもった、柔らかい
- **流音** (r): 滑らか、流れる
- **摩擦音** (sh, s): ザラザラ、シャープ

#### 特殊音素

**促音（っ）**:
- 音の切断を表現
- アタックの強調

**撥音（ん）**:
- 音の終わりを表現
- サステインの延長

**長音（ー）**:
- 音の延長を表現
- 持続時間の増加

#### 高次特徴量

**濁音の有無**:
- 0: 清音のみ → 明るい、軽い
- 1: 濁音あり → 暗い、重い

**繰り返しパターン**:
- 0: 繰り返しなし → 単発
- 1: 繰り返しあり → 連続、リズミカル

### 6.3 差分特徴量の解釈

```python
source = "カッ"  # 清音・促音
target = "ガッ"  # 濁音・促音

差分:
  consonant_g_count: +1  # 濁音が追加
  consonant_k_count: -1  # 清音が削除
  has_dakuon: +1         # 濁音フラグが立つ

→ 低音を強調、音量を上げる変換
```

---

## 7. DSPパラメータの詳細

### 7.1 パラメータ一覧

| パラメータ | 範囲 | 単位 | 効果 |
|-----------|------|------|------|
| gain_db | -24 〜 +24 | dB | 全体の音量 |
| compression | -2 〜 +2 | - | 音量の圧縮 |
| eq_sub_db | -24 〜 +24 | dB | 超低音 (60Hz) |
| eq_low_db | -24 〜 +24 | dB | 低音 (250Hz) |
| eq_mid_db | -24 〜 +24 | dB | 中音 (1kHz) |
| eq_high_db | -24 〜 +24 | dB | 高音 (4kHz) |
| eq_presence_db | -24 〜 +24 | dB | 超高音 (8kHz) |
| transient_attack | -2 〜 +2 | - | アタック強度 |
| transient_sustain | -2 〜 +2 | - | サステイン長 |
| time_stretch_ratio | 0.25 〜 2.0 | 倍率 | 再生速度 |

### 7.2 パラメータの組み合わせ効果

#### 重厚な音を作る
```python
gain_db: +15〜+20
eq_sub_db: +20〜+24
eq_low_db: +20〜+24
eq_high_db: -10〜-20
eq_presence_db: -15〜-24
time_stretch_ratio: 0.8〜0.95
```

#### 軽やかな音を作る
```python
gain_db: -5〜-15
eq_sub_db: -15〜-24
eq_low_db: -10〜-20
eq_high_db: +5〜+15
eq_presence_db: +10〜+20
time_stretch_ratio: 1.05〜1.2
```

#### 鋭い音を作る
```python
transient_attack: +1.5〜+2.0
transient_sustain: -1.5〜-2.0
eq_high_db: +10〜+20
compression: +1.0〜+2.0
```

### 7.3 デシベル (dB) の理解

```
dB値と音量比:
  +20 dB = 10倍の音量
  +12 dB ≈ 4倍の音量
  +6 dB = 2倍の音量
  0 dB = 変更なし
  -6 dB = 1/2の音量
  -12 dB ≈ 1/4の音量
  -20 dB = 1/10の音量

変換式:
  linear_gain = 10^(dB / 20)
```

---

## 8. モデル構造

### 8.1 MLPアーキテクチャ詳細

```
┌────────────────────────────────┐
│ Input Layer                     │
│ Shape: (batch_size, 38)         │
│ Description: 特徴量差分ベクトル   │
└────────────────────────────────┘
              ↓
┌────────────────────────────────┐
│ Linear Layer 1                  │
│ In: 38, Out: 32                 │
│ Weights: 38×32 = 1,216         │
│ Bias: 32                        │
│ Total: 1,248 params             │
└────────────────────────────────┘
              ↓
┌────────────────────────────────┐
│ ReLU Activation                 │
│ f(x) = max(0, x)                │
└────────────────────────────────┘
              ↓
┌────────────────────────────────┐
│ Linear Layer 2                  │
│ In: 32, Out: 10                 │
│ Weights: 32×10 = 320           │
│ Bias: 10                        │
│ Total: 330 params               │
└────────────────────────────────┘
              ↓
┌────────────────────────────────┐
│ Tanh Activation                 │
│ f(x) = (e^x - e^-x)/(e^x + e^-x)│
│ Output range: [-1, +1]          │
└────────────────────────────────┘
              ↓
┌────────────────────────────────┐
│ Output Layer                    │
│ Shape: (batch_size, 10)         │
│ Description: 正規化DSPパラメータ │
└────────────────────────────────┘
```

### 8.2 活性化関数の選択理由

**ReLU (隠れ層)**:
- 計算が高速
- 勾配消失問題を軽減
- スパース性を促進

**Tanh (出力層)**:
- 出力を-1〜+1に制限
- DSPパラメータの正規化に適している
- 正負両方向の変化を表現可能

### 8.3 モデルサイズの選択理由

**隠れ層32ユニット**:
- 38次元入力に対して適度なサイズ
- 過学習を防ぐ
- 推論速度が高速

**出力10ユニット**:
- 10個のDSPパラメータに対応
- 必要十分な表現力

---

## 9. 学習プロセス

### 9.1 データセット

**RWCP-SSD-Onomatopoeia**:
- サンプル数: 8,542
- オノマトペ種類: 1,811
- 音源カテゴリ:
  - a1-a4: 木材、金属、布等
  - b1-b5: 粒子、水、風等
  - c1-c4: ベル、コイン、ガラス、ブザー等

**データ構造**:
```csv
audio_path, onomatopoeia, confidence, avg_acceptability, num_evaluators
RWCP-SSD/drysrc/a1/cherry1/043.wav, コッ, 5, 5.0, 3
RWCP-SSD/drysrc/c1/bell2/033.wav, チリンリリン, 5, 5.0, 3
```

### 9.2 訓練データ生成

**ヒューリスティックなDSPパラメータ生成**:

`src/utils/create_rwcp_dataset.py`で実装:

```python
def generate_dsp_template(onomatopoeia, phonemes, moras):
    """
    オノマトペの音韻特徴からDSPパラメータを推定
    """
    # 濁音チェック
    has_dakuon = any(p in ['g','z','d','b'] for p in phonemes)

    # 促音チェック
    has_sokuon = 'Q' in phonemes

    # 長音チェック
    has_chouon = 'H' in phonemes

    # パラメータ初期化
    params = {
        'gain': 0.0,
        'compression': 0.0,
        'eq_sub': 0.0,
        # ...
    }

    # 濁音 → 低音強調
    if has_dakuon:
        params['eq_sub'] += 0.3
        params['eq_low'] += 0.4
        params['eq_high'] -= 0.2
        params['gain'] += 0.2

    # 促音 → アタック強調
    if has_sokuon:
        params['transient_attack'] += 0.5
        params['transient_sustain'] -= 0.3

    # 長音 → サステイン延長
    if has_chouon:
        params['transient_sustain'] += 0.4
        params['time_stretch'] += 0.1

    # 繰り返し → 音量・リズム強調
    if len(moras) >= 4 and is_repetition(moras):
        params['compression'] += 0.3
        params['gain'] += 0.15

    return params
```

### 9.3 訓練手順

**1. データセット準備**:
```bash
python src/utils/create_rwcp_dataset.py
# → data/rwcp_dataset.csv を生成
```

**2. モデル訓練**:
```bash
python src/train_with_rwcp.py
```

**訓練パラメータ**:
```python
optimizer: Adam
learning_rate: 0.001
batch_size: 32
epochs: 500
early_stopping: True (patience=50)
validation_split: 0.15
loss_function: MSELoss
```

**3. モデル保存**:
```
models/rwcp_model.pth       # モデルの重み
models/rwcp_scaler.pkl      # StandardScaler
```

### 9.4 訓練結果

```
Final Results:
  Training R²: 0.8245
  Test R²: 0.8036
  Training MSE: 0.000805
  Test MSE: 0.000900

サンプル数:
  Training: 7,260
  Validation: 1,282
  Total: 8,542
```

**性能の解釈**:
- R² = 0.8036: 約80%の分散を説明
- MSE = 0.0009: 小さい誤差（正規化空間）
- 過学習は見られない（train/testのスコアが近い）

---

## 10. ファイル構成

### 10.1 プロジェクト構造

```
Tsuji_MLP/
├── src/
│   ├── preprocessing/
│   │   ├── katakana_to_phoneme.py      # カタカナ→音素変換
│   │   ├── phoneme_to_mora.py          # 音素→モーラ変換
│   │   └── feature_extractor.py        # 特徴量抽出 (38次元)
│   │
│   ├── models/
│   │   └── mlp_model.py                # MLPモデル定義
│   │
│   ├── dsp/
│   │   └── dsp_engine.py               # DSP処理エンジン
│   │
│   ├── utils/
│   │   └── create_rwcp_dataset.py      # 訓練データ生成
│   │
│   ├── onoma2dsp.py                    # メインシステム
│   ├── train_with_rwcp.py              # 訓練スクリプト
│   ├── test_differential_onomatopoeia.py  # 差分システムテスト
│   └── test_single_differential.py     # 詳細テスト
│
├── models/
│   ├── rwcp_model.pth                  # 学習済みモデル
│   └── rwcp_scaler.pkl                 # StandardScaler
│
├── data/
│   ├── training_data_jp_utf8bom.csv    # 元データ (RWCP)
│   └── rwcp_dataset.csv                # 訓練用データ
│
├── selected_files/                     # 音声ファイル
│   ├── a1/                             # 木材系
│   ├── a2/                             # 金属系
│   ├── b1/                             # 粒子系
│   ├── c1/                             # ベル・コイン系
│   └── ...
│
├── demo_audio/
│   ├── differential_test/              # 差分システム出力
│   ├── diverse_test_amplified/         # 増幅テスト出力
│   └── ...
│
└── DIFFERENTIAL_ONOMATOPOEIA_SYSTEM.md # このドキュメント
```

### 10.2 主要ファイルの説明

#### `src/onoma2dsp.py` (350行)
**エントリーポイント**:
- `Onoma2DSP`クラス: システム全体を統合
- `process()`メソッド: 差分ベース処理の実行
- CLIインターフェース

#### `src/models/mlp_model.py` (235行)
**モデル定義**:
- `Onoma2DSPMLP`: PyTorchモデル
- `DSPParameterMapping`: パラメータマッピング
- `SklearnMLPWrapper`: scikit-learn版（未使用）

#### `src/dsp/dsp_engine.py` (約500行)
**DSP処理**:
- EQの実装
- Compressionの実装
- Transient Shaperの実装
- Time Stretchの実装
- Gainの実装

#### `src/preprocessing/feature_extractor.py` (約400行)
**特徴量抽出**:
- 38次元ベクトル生成
- 音韻統計量の計算
- 高次特徴の抽出

---

## 11. 使用方法

### 11.1 Pythonコードでの使用

**基本的な使い方**:

```python
from src.onoma2dsp import Onoma2DSP

# モデルロード
processor = Onoma2DSP(
    model_path='models/rwcp_model.pth',
    scaler_path='models/rwcp_scaler.pkl',
    amplification_factor=5.0  # 増幅係数
)

# 音声処理
result = processor.process(
    source_onomatopoeia='チリン',  # 現在の音
    target_onomatopoeia='ゴロゴロ', # 目標の音
    input_audio_path='input.wav',
    output_audio_path='output.wav',
    verbose=True  # 詳細表示
)

# 結果確認
print(f"Feature diff magnitude: {result['feature_diff_magnitude']}")
print(f"DSP params: {result['mapped_params']}")
```

### 11.2 コマンドラインでの使用

```bash
python src/onoma2dsp.py \
  --source チリン \
  --target ゴロゴロ \
  --input input.wav \
  --output output.wav \
  --model models/rwcp_model.pth \
  --scaler models/rwcp_scaler.pkl
```

**オプション**:
- `--source, -s`: 現在の音を表すオノマトペ（必須）
- `--target, -t`: 目標の音を表すオノマトペ（必須）
- `--input, -i`: 入力音声ファイル（必須）
- `--output, -p`: 出力音声ファイル（必須）
- `--model, -m`: モデルパス（デフォルト: `models/saved_model.pth`）
- `--scaler, -s`: スケーラーパス（デフォルト: `models/scaler.pkl`）
- `--sample-rate, -sr`: サンプリングレート（デフォルト: 44100）
- `--device`: デバイス（`cpu` or `cuda`）
- `--quiet, -q`: 詳細表示をオフ

### 11.3 バッチ処理の例

```python
import os
from src.onoma2dsp import Onoma2DSP

processor = Onoma2DSP(
    model_path='models/rwcp_model.pth',
    scaler_path='models/rwcp_scaler.pkl'
)

# 変換リスト
transformations = [
    ('チリン', 'ゴロゴロ'),
    ('チリン', 'ガンガン'),
    ('カラン', 'ドンドン'),
    ('カッ', 'ガッ'),
]

input_audio = 'bell.wav'
output_dir = 'processed/'

for i, (source, target) in enumerate(transformations):
    output_path = os.path.join(
        output_dir,
        f"{i:02d}_{source}_to_{target}.wav"
    )

    processor.process(
        source, target,
        input_audio, output_path,
        verbose=False
    )
    print(f"[{i+1}/{len(transformations)}] {source} -> {target}: OK")
```

### 11.4 カスタム増幅係数の使用

```python
# 控えめな変換
processor_subtle = Onoma2DSP(
    model_path='models/rwcp_model.pth',
    scaler_path='models/rwcp_scaler.pkl',
    amplification_factor=2.0  # 小さい値
)

# 劇的な変換
processor_dramatic = Onoma2DSP(
    model_path='models/rwcp_model.pth',
    scaler_path='models/rwcp_scaler.pkl',
    amplification_factor=10.0  # 大きい値
)
```

---

## 12. 実装の詳細

### 12.1 差分計算のロジック

**コード** (`src/onoma2dsp.py`):

```python
# Source特徴量抽出
source_phonemes = self.katakana_converter.convert(source_onomatopoeia)
source_moras = self.mora_converter.convert(source_phonemes)
source_features = self.feature_extractor.extract_features(
    source_phonemes, source_moras
)

# Target特徴量抽出
target_phonemes = self.katakana_converter.convert(target_onomatopoeia)
target_moras = self.mora_converter.convert(target_phonemes)
target_features = self.feature_extractor.extract_features(
    target_phonemes, target_moras
)

# 差分計算
feature_diff = target_features - source_features

# スケーリング（学習時の分布に合わせる）
if self.scaler is not None:
    feature_diff = self.scaler.transform(feature_diff.reshape(1, -1))[0]

# 差分の大きさ
magnitude = np.linalg.norm(feature_diff)
```

**差分の意味**:
- 正の値: ターゲットで増加
- 負の値: ターゲットで減少
- 大きさ: 変化の程度

### 12.2 増幅とクリッピング

**コード**:

```python
# モデル推論
with torch.no_grad():
    diff_tensor = torch.FloatTensor(feature_diff).unsqueeze(0)
    normalized_params = self.model(diff_tensor).cpu().numpy()[0]

# 増幅
normalized_params = normalized_params * self.amplification_factor

# クリッピング（-1〜+1に制限）
normalized_params = np.clip(normalized_params, -1.0, 1.0)
```

**なぜ増幅が必要か**:
- モデル出力は控えめ（-0.3〜+0.3程度）
- そのままでは変化が小さい
- 増幅により劇的な変換が可能

**クリッピングの重要性**:
- Tanhの出力範囲を超えないように
- 極端な値を防ぐ
- DSPパラメータの安定性を保つ

### 12.3 スケーラーの役割

**訓練時**:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)  # 平均・分散を学習

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 保存
import pickle
with open('models/rwcp_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

**推論時**:

```python
# ロード
with open('models/rwcp_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 適用
feature_diff_scaled = scaler.transform(feature_diff.reshape(1, -1))[0]
```

**効果**:
- 訓練データと同じ分布に正規化
- モデルの性能を最大化
- 特徴量間のスケールを統一

### 12.4 DSP処理の実装例

**EQの実装**:

```python
import scipy.signal as signal

def apply_eq_band(audio, sr, freq, gain_db, q=1.0):
    """
    特定周波数帯域にEQを適用

    Args:
        audio: 音声データ
        sr: サンプリングレート
        freq: 中心周波数 (Hz)
        gain_db: ゲイン (dB)
        q: Q値（帯域幅）
    """
    # dBを線形ゲインに変換
    gain_linear = 10 ** (gain_db / 20)

    # バターワースフィルタ設計
    nyquist = sr / 2
    low = (freq / 1.5) / nyquist
    high = (freq * 1.5) / nyquist

    # フィルタ適用
    if gain_db > 0:
        # ブースト
        sos = signal.butter(2, [low, high], btype='band', output='sos')
        filtered = signal.sosfilt(sos, audio)
        return audio + filtered * (gain_linear - 1)
    else:
        # カット
        sos = signal.butter(2, [low, high], btype='bandstop', output='sos')
        return signal.sosfilt(sos, audio) * gain_linear
```

**Time Stretchの実装**:

```python
import librosa

def apply_time_stretch(audio, sr, ratio):
    """
    再生速度を変更（ピッチ保持）

    Args:
        audio: 音声データ
        sr: サンプリングレート
        ratio: 速度倍率（0.5=半分の速さ、2.0=2倍の速さ）
    """
    # librosaのtime_stretchを使用
    stretched = librosa.effects.time_stretch(audio, rate=ratio)
    return stretched
```

---

## 13. パラメータ調整

### 13.1 amplification_factor の調整

**デフォルト値**: 5.0

**調整指針**:

| 値 | 効果 | 適用場面 |
|----|------|----------|
| 1.0 〜 3.0 | 控えめな変換 | 微調整、プレビュー |
| 3.0 〜 5.0 | バランスの取れた変換 | 通常使用 |
| 5.0 〜 8.0 | 劇的な変換 | 明確な差異が欲しい |
| 8.0 〜 10.0 | 極端な変換 | 実験的使用 |

**設定方法**:

```python
processor = Onoma2DSP(
    model_path='models/rwcp_model.pth',
    scaler_path='models/rwcp_scaler.pkl',
    amplification_factor=7.0  # カスタム値
)
```

### 13.2 DSPパラメータの手動調整

処理結果が期待通りでない場合、DSPパラメータを手動で調整可能:

```python
# 自動生成されたパラメータを取得
result = processor.process(source, target, input_audio, 'temp.wav')
params = result['mapped_params']

# パラメータを調整
params['gain_db'] *= 1.5  # ゲインを1.5倍
params['eq_low_db'] += 5  # 低音を+5dB
params['time_stretch_ratio'] = 1.0  # 速度変更を無効化

# 手動でDSP適用
from src.dsp.dsp_engine import DSPEngine
dsp = DSPEngine(sample_rate=44100)
dsp.process_audio_file(input_audio, output_audio, params)
```

### 13.3 モデルの再訓練

パラメータ生成を改善したい場合:

**1. ヒューリスティックルールを調整**:

`src/utils/create_rwcp_dataset.py`の`generate_dsp_template()`を編集:

```python
# 濁音の効果を強化
if has_dakuon:
    params['eq_sub'] += 0.5  # 0.3 → 0.5に増加
    params['eq_low'] += 0.6  # 0.4 → 0.6に増加
```

**2. データセット再生成**:

```bash
python src/utils/create_rwcp_dataset.py
```

**3. モデル再訓練**:

```bash
python src/train_with_rwcp.py
```

---

## 14. 制約事項と課題

### 14.1 現在の制約

**1. オノマトペの制約**:
- カタカナのみ対応
- ひらがな、漢字は未対応
- 未知のオノマトペでは精度が下がる可能性

**2. 音声の制約**:
- WAVファイルのみ対応
- サンプリングレート: 推奨44.1kHz
- モノラル推奨（ステレオも可能だが、両チャンネル同じ処理）

**3. モデルの制約**:
- 学習データ: RWCP-SSD-Onomatopoeiaのみ
- 音源タイプ: 打楽器・楽器音が中心
- 音声（人の声）には未対応

**4. 処理時間**:
- Time Stretchが遅い（数秒〜数十秒）
- リアルタイム処理は困難

### 14.2 既知の問題

**1. 極端なパラメータ**:
- 増幅係数が大きすぎるとクリッピングが発生
- Time stretchで0.25倍や2.0倍は音質劣化が大きい

**2. 特定のオノマトペ組み合わせ**:
- 似たオノマトペ同士（"カッ" → "カン"）では変化が小さい
- 特徴量差分が小さいため

**3. モデルの汎化性能**:
- 訓練データにないタイプの音では精度低下
- 例: 電子音、ノイズ、自然音等

### 14.3 今後の改善案

**短期的**:
1. 増幅係数の自動調整
2. より多様な音源でのテスト
3. パラメータの可視化ツール
4. バッチ処理の最適化

**中期的**:
1. より大規模なデータセットでの再訓練
2. モデルアーキテクチャの改善（より深いネットワーク）
3. オノマトペの類似度を考慮した処理
4. リアルタイム処理への対応

**長期的**:
1. エンドツーエンド学習（特徴量を学習）
2. 音声（人の声）への対応
3. より複雑なDSP処理（リバーブ、ディレイ等）
4. ユーザーフィードバックによる強化学習

---

## 15. 今後の拡張性

### 15.1 新しいDSPエフェクトの追加

現在のシステムは拡張可能な設計:

```python
# src/dsp/dsp_engine.py に追加

def apply_reverb(self, audio, reverb_amount):
    """
    リバーブエフェクトを追加
    """
    # 実装...
    pass

def apply_chorus(self, audio, depth, rate):
    """
    コーラスエフェクトを追加
    """
    # 実装...
    pass
```

**必要な変更**:
1. DSPパラメータを10次元 → N次元に拡張
2. モデル出力層を変更
3. 訓練データに新パラメータを追加
4. 再訓練

### 15.2 異なるモデルアーキテクチャ

**Transformer**:
```python
# シーケンシャルな特徴を捉える
# オノマトペの文脈を考慮

class OnomatopoeiaTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(38, 128)
        self.transformer = nn.TransformerEncoder(...)
        self.fc = nn.Linear(128, 10)
```

**CNN**:
```python
# 局所的なパターンを検出
# 音韻の組み合わせパターンを学習

class OnomatopoeiaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc = nn.Linear(64, 10)
```

### 15.3 マルチモーダル拡張

**音声 + オノマトペ**:
```python
# 入力音声の特徴も考慮
class MultimodalOnoma2DSP:
    def process(self, source_onoma, target_onoma,
                input_audio, output_audio):
        # オノマトペ特徴
        onoma_diff = self.extract_onoma_diff(source, target)

        # 音声特徴
        audio_features = self.extract_audio_features(input_audio)

        # 統合
        combined = np.concatenate([onoma_diff, audio_features])

        # 推論
        params = self.model(combined)
```

### 15.4 インタラクティブシステム

**Webアプリケーション**:
```
┌────────────────────────────────────┐
│ オノマトペ差分システム Web UI        │
├────────────────────────────────────┤
│                                    │
│ 音声アップロード: [ファイル選択]     │
│                                    │
│ 現在の音: [チリン ▼]               │
│ 目標の音: [ゴロゴロ ▼]             │
│                                    │
│ 増幅係数: [━━●━━━━] 5.0           │
│                                    │
│      [変換実行]                     │
│                                    │
│ 変換結果:                           │
│ ┌──────────────────┐              │
│ │ 🔊 元の音                         │
│ │ 🔊 変換後の音                     │
│ │                                  │
│ │ パラメータ詳細:                   │
│ │ - Gain: +21.4 dB                │
│ │ - EQ Low: +24.0 dB              │
│ │ - ...                           │
│ └──────────────────┘              │
│                                    │
│      [ダウンロード]                 │
└────────────────────────────────────┘
```

**実装技術**:
- Flask / FastAPI (バックエンド)
- React / Vue.js (フロントエンド)
- Web Audio API (音声再生)

### 15.5 リアルタイム処理

**VST/AUプラグイン化**:
```
DAW (Digital Audio Workstation)
   ↓
Onoma2DSP Plugin
   ├─ Source Onomatopoeia: [入力]
   ├─ Target Onomatopoeia: [入力]
   ├─ Amplification: [スライダー]
   └─ [Process]
   ↓
リアルタイム音声変換
```

**技術的課題**:
- Time Stretchの高速化
- レイテンシーの削減
- C++への移植

---

## まとめ

本ドキュメントでは、**差分オノマトペシステム**の全体像を詳細に説明しました。

### 主要なポイント

1. **差分ベースアプローチ**: 2つのオノマトペの特徴量差分を使用
2. **38次元特徴量**: 音韻・モーラ構造を数値化
3. **MLPモデル**: 38 → 32 → 10のシンプルな構造
4. **10次元DSPパラメータ**: EQ, compression, transient, etc.
5. **RWCP-SSDデータセット**: 8,542サンプルで訓練
6. **増幅係数**: 変換の強度を制御

### 実用性

- **直感的**: 「チリンをゴロゴロのような音にする」
- **柔軟性**: 同じターゲットでもソースにより変化
- **実用的**: 実音声にDSP処理を適用
- **拡張可能**: 新しいエフェクト・モデルを追加可能

### 次のステップ

このシステムを理解した上で：
1. 実際に使ってみる
2. パラメータを調整する
3. 新しいオノマトペを試す
4. システムを拡張する

本ドキュメントが、差分オノマトペシステムの理解と活用に役立つことを願います。

---

**ドキュメントバージョン**: 1.0
**最終更新日**: 2025年12月2日
**作成者**: Tsuji MLP Project
