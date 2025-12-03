# Onoma2DSP CLI - クイックスタート

オノマトペで音声を変換するコマンドラインツール

## 🚀 クイックスタート

### 1分で始める

```bash
# 基本的な使い方
python src/cli.py -i input.wav -s チリン -t ゴロゴロ
```

それだけです！変換された音声が `output/` フォルダに保存されます。

---

## 📝 3つの使い方

### 方法1: コマンドライン（推奨）

```bash
python src/cli.py -i <入力ファイル> -s <ソースオノマトペ> -t <ターゲットオノマトペ>
```

**例:**
```bash
# 高音 → 低音
python src/cli.py -i bell.wav -s チリン -t ゴロゴロ

# 軽快 → 重厚
python src/cli.py -i wood.wav -s コツコツ -t ドンドン

# 清音 → 濁音
python src/cli.py -i hit.wav -s カッ -t ガッ
```

### 方法2: 対話モード

```bash
# 対話モード起動
python src/cli.py
```

```
> process input.wav チリン ゴロゴロ
... (処理実行) ...

> process another.wav カッ ガッ
... (処理実行) ...

> history 5
... (履歴表示) ...

> quit
```

### 方法3: Pythonスクリプトから

```python
from src.onoma2dsp import Onoma2DSP

processor = Onoma2DSP(
    model_path='models/rwcp_model.pth',
    scaler_path='models/rwcp_scaler.pkl',
    amplification_factor=5.0,
    lambda_att=0.7
)

result = processor.process(
    'チリン', 'ゴロゴロ',
    'input.wav', 'output.wav'
)
```

---

## 🎛️ パラメータ調整

### Amplification Factor（効果の強度）

```bash
# 弱め（控えめな変換）
python src/cli.py -i input.wav -s チリン -t ゴロゴロ -f 3.0

# 標準（推奨）
python src/cli.py -i input.wav -s チリン -t ゴロゴロ -f 5.0

# 強め（劇的な変換）
python src/cli.py -i input.wav -s チリン -t ゴロゴロ -f 7.0
```

### Attention（注目機能）

```bash
# Attention OFF（基本的な変換）
python src/cli.py -i input.wav -s チリン -t ゴロゴロ -a 0.0

# Attention ON（自然な変換、推奨）
python src/cli.py -i input.wav -s チリン -t ゴロゴロ -a 0.7
```

---

## 📊 履歴機能

全ての処理は自動的に記録されます。

```bash
# 履歴を見る
python src/cli.py --history 10

# 検索する
python src/cli.py --search "チリン"
```

履歴は `history/edit_history.json` に保存されます。

---

## 💡 よく使うオノマトペ

### 高さ（周波数）
- **高音系**: チリン、キーン、ピンポン、キラキラ
- **低音系**: ゴロゴロ、ドーン、ガンガン、ズンズン

### 音色
- **鋭い**: カッ、キッ、ピッ
- **鈍い**: ガッ、ゴッ、ドッ

### 質感
- **硬い**: コツコツ、カチカチ
- **柔らかい**: ポンポン、フワフワ

### 粒度
- **細かい**: サラサラ、シャラシャラ
- **粗い**: ザラザラ、ガラガラ

---

## 📚 詳細ドキュメント

- **CLI_GUIDE.md** - 詳細な使い方ガイド
- **USAGE_GUIDE.md** - システム全般の使い方
- **demo_usage.py** - 実行可能なデモスクリプト

---

## 🔧 トラブルシューティング

### Q: ファイルが見つからないエラー

```
Error: File not found: input.wav
```

**解決方法:** ファイルパスを確認してください。相対パスまたは絶対パスで指定できます。

### Q: 結果が不自然

**解決方法:** Amplification factorを下げてみてください（例: 5.0 → 3.0）

### Q: 変化が小さい

**解決方法:** Amplification factorを上げてみてください（例: 5.0 → 7.0）

---

## 🎯 使用例

```bash
# 例1: ベルの音を低く重くする
python src/cli.py -i bell.wav -s チリン -t ゴロゴロ

# 例2: 木の打音を重厚にする
python src/cli.py -i wood.wav -s コツコツ -t ドンドン

# 例3: 粒子音を粗くする
python src/cli.py -i sand.wav -s サラサラ -t ザラザラ

# 例4: 金属音を激しくする
python src/cli.py -i metal.wav -s キラキラ -t ガンガン -f 6.0

# 例5: 履歴を確認して同じ設定を再利用
python src/cli.py --history 5
python src/cli.py -i new.wav -s チリン -t ゴロゴロ -f 5.0 -a 0.7
```

---

## ⚡ パフォーマンス

- **処理速度**: 0.5秒の音声を約0.7秒で処理
- **推奨音声長**: 0.5〜3秒（効果音に最適）
- **最大音声長**: 制限なし（ただし短い音声に最適化）

---

## 📁 ファイル構成

```
Tsuji_MLP/
├── src/
│   ├── cli.py              # CLIツール（NEW!）
│   ├── onoma2dsp.py        # メインシステム
│   └── ...
├── models/
│   ├── rwcp_model.pth      # 学習済みモデル
│   └── rwcp_scaler.pkl     # スケーラー
├── history/                 # 履歴ディレクトリ（NEW!）
│   └── edit_history.json   # 編集履歴
├── output/                  # 出力ディレクトリ
│   └── *.wav               # 変換された音声
├── CLI_GUIDE.md            # CLI詳細ガイド（NEW!）
└── README_CLI.md           # このファイル（NEW!）
```

---

**今すぐ試してみましょう！**

```bash
python src/cli.py -i selected_files/c1/bell2/033.wav -s チリン -t ゴロゴロ
```
