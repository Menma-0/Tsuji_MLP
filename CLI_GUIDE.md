# Onoma2DSP CLI ツール使い方ガイド

コマンドラインから音声を編集できるCLIツールです。編集履歴の自動記録機能付き。

## 📋 目次
1. [基本的な使い方](#基本的な使い方)
2. [対話モード](#対話モード)
3. [履歴機能](#履歴機能)
4. [コマンドリファレンス](#コマンドリファレンス)
5. [履歴ファイルの形式](#履歴ファイルの形式)

---

## 基本的な使い方

### 1. **コマンドライン引数で直接処理**

```bash
# 基本形式
python src/cli.py -i <入力ファイル> -s <ソースオノマトペ> -t <ターゲットオノマトペ>

# 例: チリン → ゴロゴロ
python src/cli.py -i input.wav -s チリン -t ゴロゴロ

# 出力ファイルを指定
python src/cli.py -i input.wav -s チリン -t ゴロゴロ -o output.wav

# パラメータ調整
python src/cli.py -i input.wav -s チリン -t ゴロゴロ -f 7.0 -a 0.5
```

### 2. **対話モードで使う**

```bash
# 対話モード起動
python src/cli.py
```

---

## 対話モード

### 起動

```bash
python src/cli.py
```

### 使用可能なコマンド

| コマンド | 説明 | 使用例 |
|---------|------|--------|
| `process` | 音声を処理 | `process input.wav チリン ゴロゴロ` |
| `set` | 設定を変更 | `set amplification_factor 7.0` |
| `settings` | 現在の設定を表示 | `settings` |
| `history` | 履歴を表示 | `history 10` |
| `search` | 履歴を検索 | `search チリン` |
| `help` | ヘルプを表示 | `help` |
| `quit/exit` | 終了 | `quit` |

### 対話モードの例

```
> process selected_files/c1/bell2/033.wav チリン ゴロゴロ

Processing:
  Input:  selected_files/c1/bell2/033.wav
  Source: チリン
  Target: ゴロゴロ
  Output: output/033_チリン_to_ゴロゴロ_20251203_133932.wav

... (処理の詳細) ...

[Success] Saved to: output/033_チリン_to_ゴロゴロ_20251203_133932.wav
[History] Entry #1 recorded

> set amplification_factor 7.0
Updated amplification_factor: 7.0

> settings

Current Settings:
  Model:               models/rwcp_model.pth
  Scaler:              models/rwcp_scaler.pkl
  Amplification Factor: 7.0
  Lambda (Attention):   0.7

> history 5

================================================================================
Edit History (Last 5 entries)
================================================================================

[#1] 2025-12-03T13:39:32.743658
  チリン -> ゴロゴロ
  Input:  selected_files/c1/bell2/033.wav
  Output: output/test_cli.wav
  Factor: 5.0, Lambda: 0.7

> quit
Goodbye!
```

---

## 履歴機能

### 自動記録

全ての処理は**自動的に履歴に記録**されます：
- 入力/出力ファイルパス
- ソース/ターゲットオノマトペ
- 使用したパラメータ（amplification_factor, lambda_att）
- 適用されたDSPパラメータの詳細
- タイムスタンプ

### 履歴の表示

```bash
# 最新10件を表示
python src/cli.py --history 10

# 対話モードで
> history 20
```

### 履歴の検索

```bash
# コマンドラインから
python src/cli.py --search "チリン"

# 対話モードで
> search ゴロゴロ
```

検索対象：
- ソースオノマトペ
- ターゲットオノマトペ
- 入力ファイルパス

---

## コマンドリファレンス

### コマンドライン引数

```
usage: cli.py [-h] [-i INPUT] [-s SOURCE] [-t TARGET] [-o OUTPUT]
              [-f FACTOR] [-a ATTENTION] [--history N] [--search QUERY]

オプション引数:
  -h, --help            ヘルプを表示
  -i INPUT, --input INPUT
                        入力音声ファイル
  -s SOURCE, --source SOURCE
                        ソースオノマトペ
  -t TARGET, --target TARGET
                        ターゲットオノマトペ
  -o OUTPUT, --output OUTPUT
                        出力音声ファイル（省略可）
  -f FACTOR, --factor FACTOR
                        Amplification factor（デフォルト: 5.0）
  -a ATTENTION, --attention ATTENTION
                        Lambda attention（デフォルト: 0.7）
  --history N           最新N件の履歴を表示
  --search QUERY        履歴を検索
```

### processコマンド（対話モード）

```
process <入力ファイル> <ソース> <ターゲット> [出力ファイル]
```

**例:**
```
> process input.wav チリン ゴロゴロ
> process input.wav チリン ゴロゴロ output.wav
```

出力ファイルを省略すると、自動的に以下の形式で生成されます：
```
output/<入力ファイル名>_<ソース>_to_<ターゲット>_<タイムスタンプ>.wav
```

### setコマンド（対話モード）

```
set <パラメータ名> <値>
```

**設定可能なパラメータ:**
- `amplification_factor` (float): DSP効果の強度（推奨: 3.0-7.0）
- `lambda_att` (float): Attention機能の強度（0.0-1.0、推奨: 0.5-0.7）
- `model_path` (string): モデルファイルのパス
- `scaler_path` (string): スケーラーファイルのパス

**例:**
```
> set amplification_factor 7.0
> set lambda_att 0.5
> set model_path models/custom_model.pth
```

### historyコマンド（対話モード）

```
history [件数]
```

**例:**
```
> history        # 最新10件
> history 5      # 最新5件
> history 50     # 最新50件
```

### searchコマンド（対話モード）

```
search <検索キーワード>
```

**例:**
```
> search チリン
> search bell2
> search ゴロゴロ
```

---

## 履歴ファイルの形式

履歴は `history/edit_history.json` にJSON形式で保存されます。

### ファイル構造

```json
[
  {
    "id": 1,
    "timestamp": "2025-12-03T13:39:32.743658",
    "input_audio": "selected_files/c1/bell2/033.wav",
    "source_onomatopoeia": "チリン",
    "target_onomatopoeia": "ゴロゴロ",
    "output_audio": "output/test_cli.wav",
    "amplification_factor": 5.0,
    "lambda_att": 0.7,
    "feature_diff_magnitude": 9.78,
    "mapped_params": {
      "gain_db": 24.0,
      "compression": 0.348,
      "eq_sub_db": 24.0,
      "eq_low_db": 24.0,
      "eq_mid_db": 0.015,
      "eq_high_db": -24.0,
      "eq_presence_db": -24.0,
      "transient_attack": -1.579,
      "transient_sustain": 0.560,
      "time_stretch_ratio": 0.893
    }
  }
]
```

### フィールド説明

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `id` | int | エントリーID（自動採番） |
| `timestamp` | string | 処理日時（ISO 8601形式） |
| `input_audio` | string | 入力ファイルパス |
| `source_onomatopoeia` | string | ソースオノマトペ |
| `target_onomatopoeia` | string | ターゲットオノマトペ |
| `output_audio` | string | 出力ファイルパス |
| `amplification_factor` | float | 使用したAmplification factor |
| `lambda_att` | float | 使用したLambda attention |
| `feature_diff_magnitude` | float | 特徴量ベクトルの差分の大きさ |
| `mapped_params` | object | 適用されたDSPパラメータの詳細 |

### 履歴ファイルの操作

履歴ファイルは標準的なJSONファイルなので、他のツールでも読み込めます：

```python
import json

# 履歴を読み込む
with open('history/edit_history.json', 'r', encoding='utf-8') as f:
    history = json.load(f)

# 分析・統計処理など
for entry in history:
    print(f"{entry['source_onomatopoeia']} -> {entry['target_onomatopoeia']}")
```

---

## 使用例

### 例1: 単発の処理

```bash
python src/cli.py -i bell.wav -s チリン -t ゴロゴロ -o bell_low.wav
```

### 例2: パラメータを変えて複数処理

```bash
# 弱め
python src/cli.py -i input.wav -s チリン -t ゴロゴロ -f 3.0 -o output_weak.wav

# 標準
python src/cli.py -i input.wav -s チリン -t ゴロゴロ -f 5.0 -o output_normal.wav

# 強め
python src/cli.py -i input.wav -s チリン -t ゴロゴロ -f 7.0 -o output_strong.wav
```

### 例3: 対話モードで連続処理

```bash
python src/cli.py
```

```
> process input1.wav チリン ゴロゴロ
> process input2.wav カッ ガッ
> process input3.wav サラサラ ザラザラ
> history 3
> quit
```

### 例4: 履歴から過去の設定を確認

```bash
# 特定のオノマトペを使った処理を検索
python src/cli.py --search "チリン"

# 結果を見て、同じ設定で別のファイルを処理
python src/cli.py -i new_input.wav -s チリン -t ゴロゴロ -f 5.0 -a 0.7
```

---

## Tips

### 💡 効率的な使い方

1. **バッチ処理**: 複数ファイルを連続処理する場合は対話モードが便利
2. **履歴活用**: 過去の処理を検索して、良かった設定を再利用
3. **設定の保存**: よく使う設定はsetコマンドで変更してから連続処理

### ⚙️ パラメータ調整のコツ

- 初回は標準設定（factor=5.0, lambda=0.7）で試す
- 効果が弱い場合はfactorを上げる（6.0-7.0）
- 効果が強すぎる場合はfactorを下げる（3.0-4.0）
- より自然な変換を求める場合はlambda_attを0.7に設定

### 📊 履歴データの活用

履歴JSONファイルは分析に利用できます：
- どのオノマトペペアが多く使われているか
- どのパラメータ設定が効果的だったか
- 特徴量差分と結果の関係性

---

**バージョン:** 1.0
**最終更新:** 2025-12-03
