# Onoma2DSP フロントエンド

Nuxt3で構築されたOnoma2DSPのWebインターフェース

## セットアップ

### 1. 依存パッケージのインストール

```bash
cd frontend
npm install
```

### 2. 開発サーバーの起動

```bash
npm run dev
```

開発サーバーが起動したら、ブラウザで http://localhost:3000 にアクセスしてください。

### 3. バックエンドAPIサーバーの起動

別のターミナルで、プロジェクトルートから以下を実行：

```bash
# FastAPIの依存パッケージをインストール（初回のみ）
pip install fastapi uvicorn[standard] python-multipart

# APIサーバーを起動
python api_server.py
```

APIサーバーは http://localhost:8000 で起動します。

- API ドキュメント: http://localhost:8000/docs
- 代替ドキュメント: http://localhost:8000/redoc

## プロジェクト構成

```
frontend/
├── app.vue              # ルートコンポーネント
├── nuxt.config.ts       # Nuxt設定
├── package.json         # 依存パッケージ
├── pages/
│   └── index.vue        # メインページ（オノマトペ入力UI）
├── assets/
│   └── css/
│       └── main.css     # グローバルスタイル
└── .nuxt/               # Nuxtビルド出力（自動生成）
```

## 使い方

1. ブラウザで http://localhost:3000 を開く
2. 音声ファイルをアップロード（WAV, MP3など）
3. 元の音を表すオノマトペを入力（例: チリン）
4. 変換後の音を表すオノマトペを入力（例: ゴロゴロ）
5. 「音声を変換」ボタンをクリック
6. 処理が完了したら、変換後の音声を再生・ダウンロード

## 詳細設定

「詳細設定」セクションで以下のパラメータを調整できます：

- **Amplification Factor** (0.0 - 2.0): モデル出力の増幅率
- **Lambda Attention** (0.0 - 1.0): Attention機能の強度

## ビルド

本番環境用にビルド：

```bash
npm run build
```

ビルド結果のプレビュー：

```bash
npm run preview
```

## トラブルシューティング

### ポート競合

開発サーバーのポートを変更する場合：

```bash
PORT=3001 npm run dev
```

### APIサーバーに接続できない

- バックエンドAPIサーバー（`python api_server.py`）が起動しているか確認
- CORS設定が正しいか確認（`api_server.py`の`allow_origins`）
- ブラウザのコンソールでエラーを確認

## API統合

フロントエンドは以下のAPIエンドポイントを使用します：

- `POST /api/process` - 音声処理
- `GET /api/history` - 処理履歴の取得
- `DELETE /api/outputs/{filename}` - 出力ファイル削除

詳細は `pages/index.vue` の `processAudio` 関数を参照してください。
