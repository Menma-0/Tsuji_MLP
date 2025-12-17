# Onoma2DSP Web UI クイックスタート

たった3ステップでWebインターフェースを起動できます！

## ステップ1: Python依存パッケージのインストール

```bash
pip install fastapi uvicorn[standard] python-multipart
```

## ステップ2: フロントエンド環境のセットアップ

```bash
cd frontend
npm install
```

## ステップ3: サーバーの起動

### ターミナル1: バックエンドAPI起動

```bash
# プロジェクトルートで実行
python api_server.py
```

### ターミナル2: フロントエンド起動

```bash
cd frontend
npm run dev
```

## アクセス

ブラウザで以下にアクセス:

- **Web UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## 使い方

1. 音声ファイルをアップロード
2. オノマトペを入力（例: チリン → ゴロゴロ）
3. 「音声を変換」をクリック
4. 結果を再生・ダウンロード

---

詳細は `WEB_SETUP.md` を参照してください。
