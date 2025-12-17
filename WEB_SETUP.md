# Onoma2DSP Web UI セットアップガイド

このガイドでは、Onoma2DSPシステムをWebインターフェースで動かすための手順を説明します。

## アーキテクチャ

```
┌─────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  Nuxt3          │         │  FastAPI         │         │  Onoma2DSP       │
│  Frontend       │ ─────→  │  API Server      │ ─────→  │  Core System     │
│  (Port 3000)    │  HTTP   │  (Port 8000)     │  Python │  (差分モデル)     │
└─────────────────┘         └──────────────────┘         └──────────────────┘
      ↓                              ↓                            ↓
  ユーザー入力              API処理・検証                   音声処理・DSP
  - オノマトペ              - CORS対応                    - 特徴量抽出
  - 音声アップロード        - ファイル管理                 - MLP予測
  - パラメータ調整          - エラーハンドリング            - DSP適用
```

## 必要な環境

- **Node.js**: v18以上
- **Python**: 3.8以上
- **npm**: v9以上
- **学習済みモデル**: `models/rwcp_model.pth`, `models/rwcp_scaler.pkl`

## セットアップ手順

### 1. バックエンド（Python）のセットアップ

#### 1-1. Python依存パッケージのインストール

```bash
# プロジェクトルートで実行
pip install -r requirements.txt
```

主な追加パッケージ：
- `fastapi`: Web APIフレームワーク
- `uvicorn`: ASGIサーバー
- `python-multipart`: ファイルアップロード対応

#### 1-2. モデルファイルの確認

以下のファイルが存在することを確認：
```bash
models/
├── rwcp_model.pth       # 学習済み差分モデル
└── rwcp_scaler.pkl      # 特徴量スケーラー
```

### 2. フロントエンド（Nuxt3）のセットアップ

#### 2-1. フロントエンドディレクトリに移動

```bash
cd frontend
```

#### 2-2. npm依存パッケージのインストール

```bash
npm install
```

インストールされる主なパッケージ：
- `nuxt`: v3.13.0以上
- `vue`: 最新版
- `@nuxt/devtools`: 開発ツール

### 3. システムの起動

#### 3-1. バックエンドAPIサーバーの起動

**ターミナル1:**

```bash
# プロジェクトルートで実行
python api_server.py
```

起動メッセージ：
```
================================================================================
Onoma2DSP API Server
================================================================================

Starting server at http://localhost:8000
API Documentation: http://localhost:8000/docs
Alternative Docs: http://localhost:8000/redoc

Press Ctrl+C to stop the server
================================================================================
```

#### 3-2. フロントエンド開発サーバーの起動

**ターミナル2:**

```bash
# frontendディレクトリで実行
cd frontend
npm run dev
```

起動メッセージ：
```
Nuxt 3.x.x with Nitro 2.x.x

  > Local:    http://localhost:3000/
  > Network:  use --host to expose
```

### 4. Webインターフェースの使用

ブラウザで **http://localhost:3000** を開きます。

#### 使用手順：

1. **音声ファイルをアップロード**
   - 対応形式: WAV, MP3, FLAC, OGG
   - クリックしてファイルを選択

2. **オノマトペを入力**
   - **元の音（Source）**: 現在の音を表すカタカナ（例: チリン）
   - **変換後の音（Target）**: 目標の音を表すカタカナ（例: ゴロゴロ）
   - カタカナ以外はエラーになります

3. **詳細設定（オプション）**
   - **Amplification Factor** (0.0 - 2.0)
     - モデル出力の増幅率
     - デフォルト: 1.0

   - **Lambda Attention** (0.0 - 1.0)
     - Attention機能の強度
     - デフォルト: 0.7

4. **「音声を変換」ボタンをクリック**
   - 処理中はボタンが「処理中...」に変わります
   - 完了まで数秒かかります

5. **結果の確認**
   - 元の音声と変換後の音声を再生可能
   - ダウンロードボタンで音声を保存

## APIエンドポイント

### POST /api/process

音声を処理します。

**リクエスト（multipart/form-data）:**
```
audio_file: File
source_onomatopoeia: string (カタカナ)
target_onomatopoeia: string (カタカナ)
amplification_factor: float (デフォルト: 1.0)
lambda_att: float (デフォルト: 0.7)
```

**レスポンス（JSON）:**
```json
{
  "status": "success",
  "source_onomatopoeia": "チリン",
  "target_onomatopoeia": "ゴロゴロ",
  "processing_time": 2.35,
  "output_filename": "20231216_120000_チリン_to_ゴロゴロ.wav",
  "output_url": "/outputs/20231216_120000_チリン_to_ゴロゴロ.wav",
  "amplification_factor": 1.0,
  "lambda_att": 0.7,
  "feature_diff_magnitude": 3.45,
  "mapped_params": {
    "gain_db": 2.5,
    "compression_ratio": 1.2,
    ...
  }
}
```

### GET /api/history

処理履歴を取得します。

**パラメータ:**
- `limit`: 取得件数（デフォルト: 10）

### DELETE /api/outputs/{filename}

出力ファイルを削除します。

## ディレクトリ構成

```
Tsuji_MLP/
├── frontend/                    # Nuxt3フロントエンド
│   ├── pages/
│   │   └── index.vue           # メインUI
│   ├── assets/
│   │   └── css/
│   │       └── main.css        # グローバルスタイル
│   ├── nuxt.config.ts          # Nuxt設定
│   ├── package.json
│   └── README.md
│
├── api_server.py               # FastAPI サーバー
├── api_uploads/                # アップロードファイル保存先（自動作成）
├── api_outputs/                # 処理済みファイル保存先（自動作成）
│
├── src/
│   ├── onoma2dsp.py            # コアシステム
│   ├── cli.py                  # CLIツール
│   ├── preprocessing/          # 前処理モジュール
│   ├── models/                 # モデル定義
│   └── dsp/                    # DSP処理
│
└── models/
    ├── rwcp_model.pth          # 学習済みモデル
    └── rwcp_scaler.pkl         # スケーラー
```

## トラブルシューティング

### 1. APIサーバーが起動しない

**エラー: `ModuleNotFoundError: No module named 'fastapi'`**

```bash
pip install fastapi uvicorn[standard] python-multipart
```

**エラー: `FileNotFoundError: models/rwcp_model.pth`**

学習済みモデルが存在するか確認：
```bash
ls models/
```

### 2. フロントエンドが起動しない

**エラー: `Cannot find module 'nuxt'`**

```bash
cd frontend
npm install
```

**ポート3000が使用中**

別のポートで起動：
```bash
PORT=3001 npm run dev
```

### 3. CORS エラー

ブラウザコンソールで `CORS policy` エラーが出る場合：

`api_server.py` の `allow_origins` を確認：
```python
allow_origins=["http://localhost:3000", "http://localhost:3001"]
```

### 4. 音声処理がエラーになる

**カタカナ検証エラー**
- オノマトペはカタカナのみ入力してください
- 半角スペースや記号は使用不可

**ファイル形式エラー**
- 対応形式: WAV, MP3, FLAC, OGG
- ファイルサイズが大きすぎる場合はエラーになる可能性があります

## 本番環境へのデプロイ

### フロントエンド

```bash
cd frontend
npm run build
npm run preview
```

### バックエンド

```bash
# Gunicornでの起動（推奨）
pip install gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 次のステップ

- [ ] ユーザー認証の追加
- [ ] 処理履歴のデータベース化
- [ ] バッチ処理機能の追加
- [ ] モデルの切り替え機能
- [ ] リアルタイム処理のサポート
- [ ] デプロイ用のDocker設定

## サポート

問題が発生した場合は、以下を確認してください：

1. ブラウザのコンソール（F12）でエラーメッセージを確認
2. APIサーバーのターミナル出力を確認
3. `api_uploads/` と `api_outputs/` ディレクトリが作成されているか確認

---

**Onoma2DSP Web Interface v1.0.0**
