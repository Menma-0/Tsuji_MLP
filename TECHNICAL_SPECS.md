# Onoma2DSP 技術仕様書

実装者向けの詳細仕様

---

## 1. システムパラメータ一覧

### 1.1 モデルパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| 入力次元 | 38 | オノマトペ特徴量差分 Δφ |
| 隠れ層次元 | 32 | MLP中間層のユニット数 |
| 出力次元 | 10 | DSPパラメータ（正規化済み） |
| 活性化関数 | ReLU (中間層), Tanh (出力層) | 非線形変換 |
| 学習率 | 0.001 | Adam optimizer |
| バッチサイズ | 32 | 学習時のミニバッチ |
| エポック数 | 200 | 学習の反復回数 |
| 損失関数 | MSE | 平均二乗誤差 |

### 1.2 DSPパラメータ範囲

| パラメータ | 正規化範囲 | 実際の範囲 | 単位 |
|-----------|-----------|-----------|------|
| gain | [-1, +1] | [-24, +24] | dB |
| compression | [-1, +1] | [-2.0, +2.0] | (無次元) |
| eq_sub | [-1, +1] | [-24, +24] | dB @ 80Hz |
| eq_low | [-1, +1] | [-24, +24] | dB @ 250Hz |
| eq_mid | [-1, +1] | [-24, +24] | dB @ 1kHz |
| eq_high | [-1, +1] | [-24, +24] | dB @ 4kHz |
| eq_presence | [-1, +1] | [-24, +24] | dB @ 10kHz |
| transient_attack | [-1, +1] | [-2.0, +2.0] | (無次元) |
| transient_sustain | [-1, +1] | [-2.0, +2.0] | (無次元) |
| time_stretch | [-1, +1] | [0.25, 2.0] | 倍率 |

### 1.3 ユーザー設定パラメータ

| パラメータ | デフォルト | 範囲 | 説明 |
|-----------|----------|------|------|
| amplification_factor | 1.0 | [0.0, 10.0] | モデル出力の増幅率 |
| lambda_att | 0.7 | [0.0, 1.0] | Attention機構の強度 |
| sample_rate | 44100 | - | サンプリングレート (Hz) |

---

## 2. データ構造

### 2.1 特徴量ベクトル（38次元）

```python
feature_vector = np.array([
    # A: 全体構造（6次元）
    M,                      # [0] モーラ数
    C_count,                # [1] 子音数
    V_count,                # [2] 母音数
    word_repeat_count,      # [3] 単語繰り返し回数
    mora_repeat_chunk_count,# [4] モーラ繰り返し塊数
    mora_repeat_ratio,      # [5] 繰り返し比率

    # B: 長さ・アクセント（4次元）
    Q_count,                # [6] 促音数
    H_mora_count,           # [7] 長音モーラ数
    H_ratio,                # [8] 長音比率
    ending_is_long,         # [9] 語末長音フラグ

    # C: 母音ヒストグラム（5次元）
    v_a_count,              # [10] 'a' 出現数
    v_i_count,              # [11] 'i' 出現数
    v_u_count,              # [12] 'u' 出現数
    v_e_count,              # [13] 'e' 出現数
    v_o_count,              # [14] 'o' 出現数

    # D: 子音カテゴリ（6次元）
    voiceless_plosive,      # [15] 無声破裂音
    voiced_plosive,         # [16] 有声破裂音
    voiceless_fric,         # [17] 無声摩擦音
    voiced_fric,            # [18] 有声摩擦音
    nasal,                  # [19] 鼻音
    approximant,            # [20] 接近音

    # E: 子音比率（3次元）
    obstruent_ratio,        # [21] 阻害音比率
    voiced_cons_ratio,      # [22] 有声子音比率
    nasal_ratio,            # [23] 鼻音比率

    # F: 位置情報（14次元）
    # 語頭子音カテゴリ（6次元ワンホット）
    first_voiceless_plosive,# [24]
    first_voiced_plosive,   # [25]
    first_voiceless_fric,   # [26]
    first_voiced_fric,      # [27]
    first_nasal,            # [28]
    first_approximant,      # [29]
    # 語末子音カテゴリ（6次元ワンホット）
    last_voiceless_plosive, # [30]
    last_voiced_plosive,    # [31]
    last_voiceless_fric,    # [32]
    last_voiced_fric,       # [33]
    last_nasal,             # [34]
    last_approximant,       # [35]
    # 母音始まり/終わり
    starts_with_vowel,      # [36]
    ends_with_vowel         # [37]
], dtype=np.float32)
```

### 2.2 子音カテゴリ定義

```python
consonant_categories = {
    'voiceless_plosive': ['p', 'py', 't', 'k', 'ky', 'ty'],
    'voiced_plosive': ['b', 'by', 'd', 'g', 'gy'],
    'voiceless_fric': ['s', 'sh', 'f', 'h', 'hy'],
    'voiced_fric': ['z', 'j'],
    'nasal': ['m', 'my', 'n', 'ny', 'N'],
    'approximant': ['r', 'ry', 'w', 'y', 'v']
}
```

### 2.3 履歴エントリ構造

```json
{
    "id": 1,
    "timestamp": "2025-12-03T13:39:32.743658",
    "input_audio": "input.wav",
    "source_onomatopoeia": "チリン",
    "target_onomatopoeia": "ゴロゴロ",
    "output_audio": "output.wav",
    "amplification_factor": 1.0,
    "lambda_att": 0.7,
    "feature_diff_magnitude": 9.784,
    "mapped_params": {
        "gain_db": 5.56,
        "compression": 0.07,
        "eq_sub_db": 7.46,
        "eq_low_db": 10.57,
        "eq_mid_db": 0.00,
        "eq_high_db": -5.00,
        "eq_presence_db": -6.93,
        "transient_attack": -0.32,
        "transient_sustain": 0.11,
        "time_stretch_ratio": 0.98
    }
}
```

---

## 3. アルゴリズム詳細

### 3.1 特徴量抽出アルゴリズム

```python
def extract_features(onomatopoeia: str) -> np.ndarray:
    """
    擬音語から38次元特徴量を抽出

    手順:
    1. カタカナ → 音素列変換
    2. 音素列 → モーラ列変換
    3. 6つのグループの特徴量を抽出
    4. 38次元ベクトルに結合
    """
    # 1. カタカナ → 音素
    phonemes = katakana_to_phoneme(onomatopoeia)
    # 例: "ガンガン" → ['g', 'a', 'N', 'g', 'a', 'N']

    # 2. 音素 → モーラ
    moras = phoneme_to_mora(phonemes)
    # 例: [('g', 'a'), ('N',), ('g', 'a'), ('N',)]

    # 3. 各グループの特徴量抽出
    features = []
    features.extend(_extract_structure_features(phonemes, moras))    # 6次元
    features.extend(_extract_length_features(phonemes, moras))       # 4次元
    features.extend(_extract_vowel_histogram(phonemes))             # 5次元
    features.extend(_extract_consonant_category_histogram(phonemes))# 6次元
    features.extend(_extract_consonant_ratio_summary(phonemes))     # 3次元
    features.extend(_extract_position_features(moras))              # 14次元

    return np.array(features, dtype=np.float32)  # 合計38次元
```

### 3.2 差分モデル推論アルゴリズム

```python
def predict_dsp_params(source_onoma: str, target_onoma: str,
                       model, scaler,
                       amplification_factor=1.0,
                       lambda_att=0.7) -> dict:
    """
    2つのオノマトペからDSPパラメータを予測

    Args:
        source_onoma: ソースオノマトペ（現在の音）
        target_onoma: ターゲットオノマトペ（目標の音）
        model: 学習済みMLPモデル
        scaler: StandardScaler（学習時のもの）
        amplification_factor: 増幅率
        lambda_att: Attention強度

    Returns:
        DSPパラメータの辞書
    """

    # ステップ1: 特徴量抽出
    φ_source = extract_features(source_onoma)  # 38次元
    φ_target = extract_features(target_onoma)  # 38次元

    # ステップ2: 差分計算
    Δφ = φ_target - φ_source  # 38次元

    # ステップ3: 標準化
    Δφ_scaled = scaler.transform(Δφ.reshape(1, -1))  # (1, 38)

    # ステップ4: モデル推論
    with torch.no_grad():
        Δφ_tensor = torch.FloatTensor(Δφ_scaled)
        ΔDSP_norm = model(Δφ_tensor).numpy()[0]  # (10,)
    # 範囲: [-1, +1]

    # ステップ5: Amplification
    ΔDSP_norm = np.clip(
        ΔDSP_norm * amplification_factor,
        -1.0, 1.0
    )

    # ステップ6: Attention補正
    if lambda_att > 0:
        # 6a. ソーステンプレート生成
        template_source = create_dsp_template(source_onoma)  # (10,)

        # 6b. 注目度ベクトル
        attention = np.abs(template_source)

        # 6c. 正規化
        max_att = np.max(attention)
        if max_att > 1e-8:
            attention = attention / max_att

        # 6d. 補正適用
        ΔDSP_norm = ΔDSP_norm * (1.0 + lambda_att * attention)
        ΔDSP_norm = np.clip(ΔDSP_norm, -1.0, 1.0)

    # ステップ7: 実際の値にマッピング
    dsp_params = map_parameters(ΔDSP_norm)

    return dsp_params
```

### 3.3 DSPパラメータマッピング

```python
def map_parameters(normalized_params: np.ndarray) -> dict:
    """
    正規化値 [-1, +1] を実際のDSPパラメータにマッピング

    変換式:
    - gain: 24.0 * x  →  [-24dB, +24dB]
    - compression: 2.0 * x  →  [-2.0, +2.0]
    - eq_*: 24.0 * x  →  [-24dB, +24dB]
    - transient_*: 2.0 * x  →  [-2.0, +2.0]
    - time_stretch: 1.0 + 0.75 * x  →  [0.25, 2.0]
    """
    assert len(normalized_params) == 10

    return {
        'gain_db': 24.0 * normalized_params[0],
        'compression': 2.0 * normalized_params[1],
        'eq_sub_db': 24.0 * normalized_params[2],
        'eq_low_db': 24.0 * normalized_params[3],
        'eq_mid_db': 24.0 * normalized_params[4],
        'eq_high_db': 24.0 * normalized_params[5],
        'eq_presence_db': 24.0 * normalized_params[6],
        'transient_attack': 2.0 * normalized_params[7],
        'transient_sustain': 2.0 * normalized_params[8],
        'time_stretch_ratio': 1.0 + 0.75 * normalized_params[9]
    }
```

### 3.4 DSPテンプレート生成（ヒューリスティック）

```python
def create_dsp_template(onomatopoeia: str) -> List[float]:
    """
    オノマトペからDSPパラメータのテンプレートを生成

    ルール:
    - 濁音（g, d, z, b）→ 低音強調、音量大
    - 高音系子音（k, p, t, s, sh, ch）→ 高音強調、音量小
    - 促音（Q）→ アタック強化
    - 長音（H）→ サスティン延長
    """
    phonemes = katakana_to_phoneme(onomatopoeia)

    # カウント
    voiced_count = sum(1 for p in phonemes if p in ['g', 'd', 'z', 'b'])
    high_consonants = sum(1 for p in phonemes if p in ['k', 'p', 't', 's', 'sh', 'ch', 'ts'])
    sokuon_count = phonemes.count('Q')
    choon_count = phonemes.count('H')

    params = [0.0] * 10

    # Gain
    if voiced_count >= 2:
        params[0] = 0.3 + 0.2 * min(voiced_count, 4)
    elif high_consonants >= 1:
        params[0] = -0.2 - 0.1 * min(high_consonants, 3)

    # Compression
    if voiced_count >= 2:
        params[1] = 0.3 + 0.1 * min(voiced_count, 4)
    elif sokuon_count > 0:
        params[1] = 0.3

    # EQ Sub
    if voiced_count >= 2:
        params[2] = 0.4 + 0.2 * min(voiced_count, 4)
    elif high_consonants >= 1:
        params[2] = -0.3 - 0.1 * min(high_consonants, 3)

    # EQ Low
    if voiced_count >= 1:
        params[3] = 0.3 + 0.2 * min(voiced_count, 4)
    elif high_consonants >= 1:
        params[3] = -0.2 - 0.1 * min(high_consonants, 3)

    # EQ Mid
    params[4] = 0.0

    # EQ High
    if high_consonants >= 1:
        params[5] = 0.4 + 0.2 * min(high_consonants, 4)
    elif voiced_count >= 2:
        params[5] = -0.2 - 0.1 * min(voiced_count, 3)

    # EQ Presence
    if high_consonants >= 1:
        params[6] = 0.5 + 0.2 * min(high_consonants, 4)
    elif voiced_count >= 2:
        params[6] = -0.2 - 0.1 * min(voiced_count, 3)

    # Transient Attack
    if sokuon_count > 0:
        params[7] = 0.7
    elif choon_count > 1:
        params[7] = -0.4

    # Transient Sustain
    if choon_count > 1:
        params[8] = 0.6
    elif sokuon_count > 0:
        params[8] = -0.3

    # Time Stretch
    if choon_count > 2:
        params[9] = 0.3
    elif sokuon_count > 1:
        params[9] = -0.2

    return params
```

---

## 4. エラーハンドリング

### 4.1 入力検証

```python
def validate_input(source_onoma: str, target_onoma: str) -> bool:
    """
    入力オノマトペの妥当性をチェック

    チェック項目:
    1. カタカナのみで構成されているか
    2. 空文字列ではないか
    3. 長すぎないか（20文字以内）
    """
    if not source_onoma or not target_onoma:
        raise ValueError("オノマトペが空です")

    if len(source_onoma) > 20 or len(target_onoma) > 20:
        raise ValueError("オノマトペが長すぎます（20文字以内）")

    import re
    katakana_pattern = r'^[ァ-ヴー]+$'
    if not re.match(katakana_pattern, source_onoma):
        raise ValueError(f"ソースオノマトペが無効です: {source_onoma}")
    if not re.match(katakana_pattern, target_onoma):
        raise ValueError(f"ターゲットオノマトペが無効です: {target_onoma}")

    return True
```

### 4.2 音声ファイル検証

```python
def validate_audio_file(audio_path: str) -> bool:
    """
    音声ファイルの妥当性をチェック

    チェック項目:
    1. ファイルが存在するか
    2. WAV形式か
    3. 読み込み可能か
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    if not audio_path.endswith('.wav'):
        raise ValueError("WAV形式のファイルのみサポートされています")

    try:
        import soundfile as sf
        data, sr = sf.read(audio_path)
    except Exception as e:
        raise ValueError(f"音声ファイルの読み込みに失敗: {e}")

    return True
```

---

## 5. パフォーマンス最適化

### 5.1 処理時間の内訳（0.5秒音声の場合）

```
合計: 約0.7秒
├─ 特徴量抽出: <0.01秒
├─ モデル推論: <0.01秒
├─ DSP処理:
│  ├─ EQ: 0.3秒
│  ├─ Compression: 0.1秒
│  ├─ Transient: 0.1秒
│  ├─ Time Stretch: 0.15秒
│  └─ Gain: <0.01秒
└─ ファイルI/O: 0.05秒
```

### 5.2 最適化のポイント

1. **モデル推論の高速化**
   - PyTorchの`torch.no_grad()`を使用
   - バッチ処理は不要（1サンプルのみ）

2. **DSP処理の最適化**
   - NumPy/SciPyのベクトル化演算
   - 不要な処理をスキップ

3. **メモリ管理**
   - 音声データは必要最小限のみロード
   - 中間バッファは適切に解放

---

## 6. テスト仕様

### 6.1 ユニットテスト

```python
def test_feature_extraction():
    """特徴量抽出のテスト"""
    onoma = "ガンガン"
    features = extract_features(onoma)

    assert features.shape == (38,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))

def test_model_inference():
    """モデル推論のテスト"""
    model = Onoma2DSPMLP(d_in=38, d_out=10, hidden_dim=32)
    dummy_input = torch.randn(1, 38)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 10)
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0)

def test_dsp_mapping():
    """DSPマッピングのテスト"""
    normalized = np.array([0.5, -0.3, 0.0, 1.0, -1.0,
                          0.8, -0.5, 0.2, -0.1, 0.0])
    params = map_parameters(normalized)

    assert 'gain_db' in params
    assert -24.0 <= params['gain_db'] <= 24.0
    assert 0.25 <= params['time_stretch_ratio'] <= 2.0
```

### 6.2 統合テスト

```python
def test_end_to_end():
    """エンドツーエンドのテスト"""
    processor = Onoma2DSP(
        model_path='models/rwcp_model.pth',
        scaler_path='models/rwcp_scaler.pkl'
    )

    result = processor.process(
        'チリン', 'ゴロゴロ',
        'test_input.wav',
        'test_output.wav'
    )

    assert os.path.exists('test_output.wav')
    assert 'mapped_params' in result
    assert len(result['mapped_params']) == 10
```

---

## 7. デプロイメント

### 7.1 必要なファイル

```
production/
├── models/
│   ├── rwcp_model.pth          # 学習済みモデル（必須）
│   └── rwcp_scaler.pkl         # StandardScaler（必須）
├── src/
│   ├── preprocessing/          # 前処理モジュール（必須）
│   ├── models/                 # モデル定義（必須）
│   ├── dsp/                    # DSP処理（必須）
│   ├── onoma2dsp.py            # メインシステム（必須）
│   └── cli.py                  # CLIインターフェース（オプション）
└── requirements.txt            # 依存パッケージ（必須）
```

### 7.2 依存パッケージ

```txt
torch>=1.10.0
numpy>=1.20.0
scipy>=1.7.0
librosa>=0.9.0
soundfile>=0.11.0
scikit-learn>=1.0.0
pandas>=1.3.0
```

### 7.3 環境変数

```bash
# オプション: モデルパスの上書き
export ONOMA2DSP_MODEL_PATH=/path/to/custom_model.pth
export ONOMA2DSP_SCALER_PATH=/path/to/custom_scaler.pkl
```

---

## 8. トラブルシューティング

### 8.1 よくある問題

**問題: モデルのロードに失敗**
```
Error: torch.load() failed
```
**解決策:**
- モデルファイルが存在するか確認
- PyTorchのバージョンを確認（>=1.10.0）
- `weights_only=False`オプションを追加

**問題: 音声処理が遅い**
```
処理に10秒以上かかる
```
**解決策:**
- 音声ファイルのサイズを確認（推奨: 0.5〜3秒）
- NumPy/SciPyが最新版か確認
- 不要なverbose出力を無効化

**問題: 結果が不自然**
```
音声が歪む、ノイズが入る
```
**解決策:**
- amplification_factorを下げる（1.0 → 0.5）
- lambda_attを調整（0.7 → 0.3）
- DSPパラメータの範囲を確認

---

**最終更新:** 2025-12-03
**バージョン:** 1.0
