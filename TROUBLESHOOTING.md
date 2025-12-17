# Onoma2DSP Web UI トラブルシューティング

## "Failed to fetch" エラーが発生する場合

### 問題の診断

このエラーは、フロントエンドがバックエンドAPIサーバーに接続できない場合に発生します。

### 解決手順

#### 1. バックエンドAPIサーバーが起動しているか確認

**ターミナル1で以下を実行:**

```bash
python api_server.py
```

**正常に起動すると以下のように表示されます:**

```
================================================================================
Onoma2DSP API Server
================================================================================

Starting server at http://localhost:8000
API Documentation: http://localhost:8000/docs
Alternative Docs: http://localhost:8000/redoc

Press Ctrl+C to stop the server
================================================================================

INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### 2. APIサーバーにアクセスできるか確認

ブラウザで以下のURLを開いてください:

- **ヘルスチェック**: http://localhost:8000
  - 正常なら: `{"status":"ok","message":"Onoma2DSP API Server is running","version":"1.0.0"}`

- **API ドキュメント**: http://localhost:8000/docs
  - SwaggerUIが表示されるはず

#### 3. フロントエンド開発サーバーを起動

**ターミナル2で以下を実行:**

```bash
cd frontend
npm run dev
```

**正常に起動すると:**

```
Nuxt 3.x.x with Nitro 2.x.x

  > Local:    http://localhost:3000/
  > Network:  use --host to expose
```

#### 4. ブラウザで接続テスト

1. ブラウザで http://localhost:3000 を開く
2. 開発者ツールを開く（F12キー）
3. コンソールタブを確認
4. 音声変換を試す
5. コンソールに以下のログが表示されるか確認:

```
[Frontend] Checking API server connection...
[Frontend] Sending request to API...
  Source: チリン
  Target: ゴロゴロ
  File: test.wav
[Frontend] Response status: 200
```

---

## よくある問題と解決策

### 問題1: `ModuleNotFoundError: No module named 'fastapi'`

**原因:** FastAPIがインストールされていない

**解決策:**
```bash
pip install fastapi uvicorn[standard] python-multipart
```

### 問題2: ポート8000が既に使用中

**エラーメッセージ:**
```
ERROR:    [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000): 通常、各ソケット アドレスに対してプロトコル、ネットワーク アドレス、またはポートのどれか 1 つのみを使用できます。
```

**解決策1: 別のポートを使用**

`api_server.py` の最後の行を変更:
```python
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8001,  # 8000 → 8001に変更
    log_level="info"
)
```

フロントエンドの `pages/index.vue` も変更:
```javascript
const response = await fetch('http://localhost:8001/api/process', {
```

**解決策2: ポートを使用しているプロセスを終了**

```bash
# Windowsの場合
netstat -ano | findstr :8000
taskkill /PID <プロセスID> /F

# Linuxの場合
lsof -i :8000
kill -9 <プロセスID>
```

### 問題3: CORSエラー

**ブラウザコンソールのエラー:**
```
Access to fetch at 'http://localhost:8000/api/process' from origin 'http://localhost:3000' has been blocked by CORS policy
```

**原因:** CORS設定が正しくない

**解決策:** すでに `api_server.py` で修正済み（`allow_origins=["*"]`）

もし問題が続く場合、以下を確認:
1. ブラウザのキャッシュをクリア
2. APIサーバーを再起動
3. ブラウザを再起動

### 問題4: フロントエンドが起動しない

**エラー: `Cannot find module 'nuxt'`**

```bash
cd frontend
npm install
```

**ポート3000が使用中:**

```bash
PORT=3001 npm run dev
```

### 問題5: 音声ファイルのアップロードエラー

**エラー: `サポートされていないファイル形式です: .xxx`**

**対応フォーマット:**
- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- OGG (.ogg)

### 問題6: カタカナ検証エラー

**エラー: `source_onomatopoeiaはカタカナのみ入力してください`**

**確認事項:**
- 全角カタカナのみ入力
- 半角カタカナは不可
- スペースや記号は不可
- 長音記号（ー）は使用可能

**正しい例:**
- チリン ✓
- ゴロゴロ ✓
- ドーン ✓

**間違った例:**
- ﾁﾘﾝ ✗（半角）
- チリン！ ✗（記号）
- chirin ✗（英字）

---

## デバッグ方法

### バックエンドのログを確認

APIサーバーのターミナルで以下のようなログが表示されます:

```
============================================================
[API] New request received
  Source: チリン
  Target: ゴロゴロ
  File: test.wav
  Amplification: 1.0
  Lambda: 0.7
============================================================
```

### フロントエンドのログを確認

ブラウザの開発者ツール（F12）のコンソールタブで:

```
[Frontend] Checking API server connection...
[Frontend] Sending request to API...
  Source: チリン
  Target: ゴロゴロ
  File: test.wav
[Frontend] Response status: 200
```

### ネットワークタブでリクエストを確認

1. 開発者ツール（F12）を開く
2. Networkタブを選択
3. 音声変換を実行
4. `process` というリクエストをクリック
5. Headers、Payload、Responseを確認

---

## 完全なリセット手順

全てがうまくいかない場合、以下の手順で環境をリセット:

### 1. 全てのサーバーを停止

- ターミナルでCtrl+Cを押す
- 全てのPythonプロセスを終了

### 2. キャッシュをクリア

```bash
# フロントエンドのキャッシュ
cd frontend
rm -rf .nuxt node_modules package-lock.json
npm install

# ブラウザのキャッシュもクリア
```

### 3. 依存パッケージを再インストール

```bash
# Python
pip uninstall fastapi uvicorn python-multipart -y
pip install fastapi uvicorn[standard] python-multipart

# Node.js
cd frontend
npm install
```

### 4. サーバーを再起動

```bash
# ターミナル1: バックエンド
python api_server.py

# ターミナル2: フロントエンド
cd frontend
npm run dev
```

---

## それでも解決しない場合

### システム情報を確認

```bash
# Python バージョン
python --version

# Node.js バージョン
node --version

# npm バージョン
npm --version

# インストール済みパッケージ
pip list | findstr "fastapi uvicorn"
```

### 必要な環境

- Python: 3.8以上
- Node.js: 18以上
- npm: 9以上

### ファイアウォール設定

Windowsファイアウォールでポート8000と3000が許可されているか確認

### ブラウザ

Chrome、Firefox、Edgeの最新版を使用することを推奨

---

## サポート

問題が解決しない場合は、以下の情報を添えて報告してください:

1. エラーメッセージの全文
2. ブラウザのコンソールログ
3. APIサーバーのログ
4. 使用しているOS
5. Python、Node.jsのバージョン
