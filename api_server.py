"""
FastAPI server for Onoma2DSP
Nuxt3フロントエンドと連携するためのAPIサーバー
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import sys
import os
import time
import shutil
import uuid
from pathlib import Path
from datetime import datetime

# プロジェクトルートを追加
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.onoma2dsp import Onoma2DSP

# FastAPIアプリ初期化
app = FastAPI(
    title="Onoma2DSP API",
    description="オノマトペで音声を編集するシステムのAPI",
    version="1.0.0"
)

# CORS設定（Nuxt3からのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発環境では全てのオリジンを許可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 作業ディレクトリの作成
UPLOAD_DIR = Path("api_uploads")
OUTPUT_DIR = Path("api_outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# セッション管理用のデータ構造
class EditStep(BaseModel):
    """編集ステップのデータ"""
    step: int                           # 0=元音声, 1-3=編集結果
    audio_path: str                     # サーバー上のファイルパス
    audio_url: str                      # クライアント用URL
    source_onoma: Optional[str] = None  # 元のオノマトペ
    target_onoma: Optional[str] = None  # 変換後のオノマトペ
    timestamp: str                      # タイムスタンプ
    processing_time: Optional[float] = None  # 処理時間
    dsp_params: Optional[Dict] = None   # このステップのDSP差分パラメータ

class EditSession(BaseModel):
    """編集セッションのデータ"""
    session_id: str
    history: List[EditStep]             # 最大4要素（元音声 + 3編集）
    current_step: int                   # 0-3
    created_at: str

# グローバルセッションストレージ（本番ではRedis等を使用）
sessions: Dict[str, EditSession] = {}

# Onoma2DSPプロセッサのグローバルインスタンス（遅延初期化）
processor = None


def get_processor():
    """プロセッサのシングルトン取得"""
    global processor
    if processor is None:
        processor = Onoma2DSP(
            model_path='models/rwcp_model.pth',
            scaler_path='models/rwcp_scaler.pkl',
            amplification_factor=1.0,
            lambda_att=0.7
        )
    return processor


def merge_dsp_params(base: Dict, delta: Dict) -> Dict:
    """
    DSPパラメータを累積する

    Args:
        base: 基準となるDSPパラメータ（累積値）
        delta: 追加するDSPパラメータ（差分）

    Returns:
        累積されたDSPパラメータ
    """
    result = base.copy() if base else {}

    if not delta:
        return result

    # 加算ベースのパラメータ
    additive_params = [
        'gain_db',
        'eq_sub_db', 'eq_low_db', 'eq_mid_db', 'eq_high_db', 'eq_presence_db',
        'compression',
        'transient_attack', 'transient_sustain'
    ]

    for param in additive_params:
        if param in delta:
            result[param] = result.get(param, 0.0) + delta[param]

    # 乗算ベースのパラメータ（time_stretch_ratio）
    if 'time_stretch_ratio' in delta:
        base_ratio = result.get('time_stretch_ratio', 1.0)
        delta_ratio = delta['time_stretch_ratio']
        # 1.0を基準とした乗算
        result['time_stretch_ratio'] = base_ratio * delta_ratio

    return result


def clip_dsp_params(params: Dict) -> Dict:
    """
    DSPパラメータを適切な範囲に制限する

    Args:
        params: DSPパラメータ

    Returns:
        制限されたDSPパラメータ
    """
    result = params.copy()

    # dBベースのパラメータ: ±30dBに制限
    db_params = [
        'gain_db',
        'eq_sub_db', 'eq_low_db', 'eq_mid_db', 'eq_high_db', 'eq_presence_db'
    ]
    for param in db_params:
        if param in result:
            result[param] = max(-30.0, min(30.0, result[param]))

    # compression: -2.0 ~ 2.0に制限
    if 'compression' in result:
        result['compression'] = max(-2.0, min(2.0, result['compression']))

    # transient: -2.0 ~ 2.0に制限
    if 'transient_attack' in result:
        result['transient_attack'] = max(-2.0, min(2.0, result['transient_attack']))
    if 'transient_sustain' in result:
        result['transient_sustain'] = max(-2.0, min(2.0, result['transient_sustain']))

    # time_stretch_ratio: 0.25 ~ 2.0に制限
    if 'time_stretch_ratio' in result:
        result['time_stretch_ratio'] = max(0.25, min(2.0, result['time_stretch_ratio']))

    return result


@app.get("/")
async def root():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "message": "Onoma2DSP API Server is running",
        "version": "1.0.0"
    }


@app.post("/api/process")
async def process_audio(
    audio_file: UploadFile = File(...),
    source_onomatopoeia: str = Form(...),
    target_onomatopoeia: str = Form(...),
    amplification_factor: float = Form(1.0),
    lambda_att: float = Form(0.7)
):
    """
    音声処理エンドポイント

    Args:
        audio_file: 入力音声ファイル
        source_onomatopoeia: 元の音を表すオノマトペ（カタカナ）
        target_onomatopoeia: 変換後の音を表すオノマトペ（カタカナ）
        amplification_factor: 増幅率（デフォルト: 1.0）
        lambda_att: Attention強度（デフォルト: 0.7）

    Returns:
        処理結果とメタデータ
    """
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"[API] New request received")
    print(f"  Source: {source_onomatopoeia}")
    print(f"  Target: {target_onomatopoeia}")
    print(f"  File: {audio_file.filename if audio_file else 'None'}")
    print(f"  Amplification: {amplification_factor}")
    print(f"  Lambda: {lambda_att}")
    print(f"{'='*60}\n")

    try:
        # 1. ファイル検証
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が無効です")

        # サポートされる拡張子チェック
        ext = Path(audio_file.filename).suffix.lower()
        if ext not in ['.wav', '.mp3', '.flac', '.ogg']:
            raise HTTPException(
                status_code=400,
                detail=f"サポートされていないファイル形式です: {ext}"
            )

        # 2. カタカナ検証
        import re
        katakana_pattern = re.compile(r'^[ァ-ヶー]+$')
        if not katakana_pattern.match(source_onomatopoeia):
            raise HTTPException(
                status_code=400,
                detail="source_onomatopoeiaはカタカナのみ入力してください"
            )
        if not katakana_pattern.match(target_onomatopoeia):
            raise HTTPException(
                status_code=400,
                detail="target_onomatopoeiaはカタカナのみ入力してください"
            )

        # 3. 一時ファイル保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"{timestamp}_{audio_file.filename}"
        input_path = UPLOAD_DIR / input_filename

        with open(input_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        # 4. 出力ファイルパス
        output_filename = f"{timestamp}_{source_onomatopoeia}_to_{target_onomatopoeia}.wav"
        output_path = OUTPUT_DIR / output_filename

        # 5. プロセッサ初期化（パラメータ更新）
        proc = Onoma2DSP(
            model_path='models/rwcp_model.pth',
            scaler_path='models/rwcp_scaler.pkl',
            amplification_factor=amplification_factor,
            lambda_att=lambda_att
        )

        # 6. 音声処理実行
        result = proc.process(
            source_onomatopoeia=source_onomatopoeia,
            target_onomatopoeia=target_onomatopoeia,
            input_audio_path=str(input_path),
            output_audio_path=str(output_path),
            verbose=True
        )

        # 7. 処理時間計算
        processing_time = time.time() - start_time

        # 8. レスポンス作成
        response = {
            "status": "success",
            "source_onomatopoeia": source_onomatopoeia,
            "target_onomatopoeia": target_onomatopoeia,
            "processing_time": round(processing_time, 2),
            "input_filename": audio_file.filename,
            "output_filename": output_filename,
            "output_url": f"/outputs/{output_filename}",
            "amplification_factor": amplification_factor,
            "lambda_att": lambda_att,
            "feature_diff_magnitude": result.get('feature_diff_magnitude', 0),
            "mapped_params": result.get('mapped_params', {})
        }

        # 9. 入力ファイルを削除（オプション）
        # input_path.unlink()

        return response

    except Exception as e:
        # エラーハンドリング
        import traceback
        error_detail = {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"Error processing audio: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history(limit: int = 10):
    """
    処理履歴を取得

    Args:
        limit: 取得する履歴の最大数

    Returns:
        処理履歴のリスト
    """
    try:
        from src.cli import HistoryManager

        history_manager = HistoryManager()
        history = history_manager.get_history(limit=limit)

        return {
            "status": "success",
            "count": len(history),
            "history": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/outputs/{filename}")
async def delete_output(filename: str):
    """
    出力ファイルを削除

    Args:
        filename: 削除するファイル名

    Returns:
        削除結果
    """
    try:
        file_path = OUTPUT_DIR / filename
        if file_path.exists():
            file_path.unlink()
            return {
                "status": "success",
                "message": f"ファイル {filename} を削除しました"
            }
        else:
            raise HTTPException(status_code=404, detail="ファイルが見つかりません")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== セッション管理エンドポイント =====

@app.post("/api/session/create")
async def create_session(audio_file: UploadFile = File(...)):
    """
    新しい編集セッションを作成
    元音声をアップロードしてsession_idを返す

    Args:
        audio_file: 元音声ファイル

    Returns:
        session_id: セッションID
        audio_url: 元音声のURL
    """
    try:
        # セッションID生成
        session_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # ファイル検証
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="ファイル名が無効です")

        ext = Path(audio_file.filename).suffix.lower()
        if ext not in ['.wav', '.mp3', '.flac', '.ogg']:
            raise HTTPException(
                status_code=400,
                detail=f"サポートされていないファイル形式です: {ext}"
            )

        # 元音声を保存
        original_filename = f"{session_id}_original{ext}"
        original_path = OUTPUT_DIR / original_filename

        with open(original_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        # セッション作成
        session = EditSession(
            session_id=session_id,
            history=[
                EditStep(
                    step=0,
                    audio_path=str(original_path),
                    audio_url=f"/outputs/{original_filename}",
                    source_onoma=None,
                    target_onoma=None,
                    timestamp=timestamp.isoformat(),
                    processing_time=None,
                    dsp_params=None  # 元音声にはパラメータなし
                )
            ],
            current_step=0,
            created_at=timestamp.isoformat()
        )

        sessions[session_id] = session

        print(f"\n[Session Created] ID: {session_id}")
        print(f"  Original file: {original_filename}")
        print(f"  Total sessions: {len(sessions)}\n")

        return {
            "session_id": session_id,
            "audio_url": f"/outputs/{original_filename}"
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error creating session: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/{session_id}/edit")
async def edit_session(
    session_id: str,
    source_onomatopoeia: str = Form(...),
    target_onomatopoeia: str = Form(...),
    amplification_factor: float = Form(1.0),
    lambda_att: float = Form(0.7)
):
    """
    セッションに編集を追加（最大3回まで）
    前回の音声を入力として使用

    Args:
        session_id: セッションID
        source_onomatopoeia: 元の音を表すオノマトペ（カタカナ）
        target_onomatopoeia: 変換後の音を表すオノマトペ（カタカナ）
        amplification_factor: 増幅率（デフォルト: 1.0）
        lambda_att: Attention強度（デフォルト: 0.7）

    Returns:
        session_id: セッションID
        current_step: 現在の編集回数
        audio_url: 新しい音声のURL
        processing_time: 処理時間
        history: 全編集履歴
    """
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"[Session Edit] ID: {session_id}")
    print(f"  Source: {source_onomatopoeia}")
    print(f"  Target: {target_onomatopoeia}")
    print(f"  Amplification: {amplification_factor}")
    print(f"  Lambda: {lambda_att}")
    print(f"{'='*60}\n")

    try:
        # セッション確認
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = sessions[session_id]

        # 最大編集回数チェック
        if session.current_step >= 3:
            raise HTTPException(status_code=400, detail="Maximum edits (3) reached")

        # カタカナ検証
        import re
        katakana_pattern = re.compile(r'^[ァ-ヶー]+$')
        if not katakana_pattern.match(source_onomatopoeia):
            raise HTTPException(
                status_code=400,
                detail="source_onomatopoeiaはカタカナのみ入力してください"
            )
        if not katakana_pattern.match(target_onomatopoeia):
            raise HTTPException(
                status_code=400,
                detail="target_onomatopoeiaはカタカナのみ入力してください"
            )

        # ★ 常に元音声を使用（音質劣化防止）
        original_audio_path = session.history[0].audio_path

        # 出力パス
        output_filename = f"{session_id}_edit{session.current_step + 1}.wav"
        output_path = OUTPUT_DIR / output_filename

        # プロセッサ初期化
        proc = Onoma2DSP(
            model_path='models/rwcp_model.pth',
            scaler_path='models/rwcp_scaler.pkl',
            amplification_factor=amplification_factor,
            lambda_att=lambda_att
        )

        # ★ 1. 新しい編集のDSPパラメータを計算（音声処理なし）
        new_dsp_params = proc.predict_dsp_params(
            source_onomatopoeia=source_onomatopoeia,
            target_onomatopoeia=target_onomatopoeia,
            verbose=True
        )

        # ★ 2. これまでのDSPパラメータを累積
        cumulative_params = {}
        for past_edit in session.history[1:]:  # step 1以降
            if past_edit.dsp_params:
                cumulative_params = merge_dsp_params(cumulative_params, past_edit.dsp_params)

        # ★ 3. 新しいパラメータを追加
        final_params = merge_dsp_params(cumulative_params, new_dsp_params)

        # ★ 4. パラメータを制限
        final_params = clip_dsp_params(final_params)

        print(f"\n[DSP Parameter Accumulation]")
        print(f"  New params: {new_dsp_params}")
        print(f"  Cumulative params: {cumulative_params}")
        print(f"  Final params (clipped): {final_params}\n")

        # ★ 5. 元音声に累積パラメータを適用
        proc.apply_dsp_only(
            input_audio_path=original_audio_path,  # 常に元音声
            output_audio_path=str(output_path),
            dsp_params=final_params,  # 累積されたパラメータ
            verbose=True
        )

        processing_time = time.time() - start_time

        # セッションに編集を追加（差分パラメータを保存）
        new_step = EditStep(
            step=session.current_step + 1,
            audio_path=str(output_path),
            audio_url=f"/outputs/{output_filename}",
            source_onoma=source_onomatopoeia,
            target_onoma=target_onomatopoeia,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            dsp_params=new_dsp_params  # ★ 差分パラメータを保存
        )

        session.history.append(new_step)
        session.current_step += 1

        print(f"[Session Updated] ID: {session_id}")
        print(f"  Current step: {session.current_step}/3")
        print(f"  Processing time: {processing_time:.2f}s\n")

        return {
            "session_id": session_id,
            "current_step": session.current_step,
            "audio_url": f"/outputs/{output_filename}",
            "processing_time": round(processing_time, 2),
            "history": [step.dict() for step in session.history]
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error editing session: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """
    セッション情報を取得

    Args:
        session_id: セッションID

    Returns:
        セッション情報
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = sessions[session_id]
        return {
            "session_id": session.session_id,
            "current_step": session.current_step,
            "created_at": session.created_at,
            "history": [step.dict() for step in session.history]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """
    セッションと関連ファイルを削除

    Args:
        session_id: セッションID

    Returns:
        削除結果
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = sessions[session_id]

        # ファイル削除
        deleted_files = []
        for step in session.history:
            audio_path = Path(step.audio_path)
            if audio_path.exists():
                audio_path.unlink()
                deleted_files.append(audio_path.name)

        # セッション削除
        del sessions[session_id]

        print(f"[Session Deleted] ID: {session_id}")
        print(f"  Files deleted: {len(deleted_files)}")
        print(f"  Total sessions: {len(sessions)}\n")

        return {
            "status": "deleted",
            "session_id": session_id,
            "deleted_files": deleted_files
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error deleting session: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# 静的ファイルの提供（処理済み音声ファイル）
# 注: これは全てのAPIルートの後に配置する必要があります
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


if __name__ == "__main__":
    print("=" * 80)
    print("Onoma2DSP API Server")
    print("=" * 80)
    print("\nStarting server at http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
