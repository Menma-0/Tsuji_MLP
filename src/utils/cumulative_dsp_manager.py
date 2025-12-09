"""
累積DSPパラメータ管理システム

問題:
- 複数回の編集を重ねると音声が劣化する（音量低下、周波数帯域の縮小）
- DSPフィルタを重ねがけすると品質が落ちる

解決策:
方針A: 累積パラメータを管理し、常に元音から再レンダリング
方針B: パラメータ上下限と減速ロジック

Usage:
    manager = CumulativeDSPManager()
    manager.set_original_audio("input.wav")

    # 1回目の編集
    dsp_diff = model_output  # モデルの出力
    effective_diff = manager.apply_deceleration(dsp_diff)  # 減速適用
    manager.update_parameters(effective_diff)  # 累積パラメータを更新
    output_audio = manager.render("output1.wav")  # 元音から再レンダリング

    # 2回目の編集（続き）
    dsp_diff2 = model_output2
    effective_diff2 = manager.apply_deceleration(dsp_diff2)
    manager.update_parameters(effective_diff2)
    output_audio = manager.render("output2.wav")  # 元音から累積パラメータで再レンダリング
"""
import numpy as np
import os
import json
import shutil
from datetime import datetime
from typing import Dict, Optional, Tuple

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dsp.dsp_engine import DSPEngine
from src.models.mlp_model import DSPParameterMapping


class CumulativeDSPManager:
    """累積DSPパラメータを管理し、元音からの再レンダリングを行うクラス"""

    # DSPパラメータのインデックスと名前
    PARAM_NAMES = [
        'gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
        'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch'
    ]

    # 各パラメータの上下限（正規化後の値: -1〜+1の範囲での限界）
    # これを超えると音質劣化が顕著になる
    PARAM_LIMITS = {
        'gain': (-0.8, 0.8),           # -19.2dB 〜 +19.2dB (24dB * 0.8)
        'compression': (-0.7, 0.7),     # 圧縮/拡張の限界
        'eq_sub': (-0.5, 0.5),          # -12dB 〜 +12dB (24dB * 0.5)
        'eq_low': (-0.5, 0.5),          # -12dB 〜 +12dB
        'eq_mid': (-0.6, 0.6),          # -14.4dB 〜 +14.4dB
        'eq_high': (-0.5, 0.5),         # -12dB 〜 +12dB
        'eq_presence': (-0.5, 0.5),     # -12dB 〜 +12dB
        'attack': (-0.8, 0.8),          # トランジェント調整限界
        'sustain': (-0.8, 0.8),         # トランジェント調整限界
        'time_stretch': (-0.5, 0.5)     # 0.625x 〜 1.375x
    }

    def __init__(self, sample_rate: int = 44100, state_file: str = None):
        """
        Args:
            sample_rate: サンプリングレート
            state_file: 状態を保存するファイルパス（Noneで自動生成）
        """
        self.sample_rate = sample_rate
        self.dsp_engine = DSPEngine(sample_rate=sample_rate)
        self.mapper = DSPParameterMapping()

        # 状態管理
        self.state_file = state_file or 'history/cumulative_dsp_state.json'
        self.state_dir = os.path.dirname(self.state_file)
        if self.state_dir:
            os.makedirs(self.state_dir, exist_ok=True)

        # 累積パラメータ（10次元、初期値0）
        self.cumulative_params = np.zeros(10)

        # 元音声のパス（コピーを保持）
        self.original_audio_path = None
        self.original_audio_backup_path = None

        # 編集履歴（パラメータ変化の追跡用）
        self.edit_history = []

        # 状態をロード（存在する場合）
        self._load_state()

    def _load_state(self):
        """保存された状態をロード"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.cumulative_params = np.array(state.get('cumulative_params', np.zeros(10)))
                    self.original_audio_path = state.get('original_audio_path')
                    self.original_audio_backup_path = state.get('original_audio_backup_path')
                    self.edit_history = state.get('edit_history', [])
            except Exception as e:
                print(f"Warning: Could not load state: {e}")

    def _save_state(self):
        """状態を保存"""
        state = {
            'cumulative_params': self.cumulative_params.tolist(),
            'original_audio_path': self.original_audio_path,
            'original_audio_backup_path': self.original_audio_backup_path,
            'edit_history': self.edit_history,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def set_original_audio(self, audio_path: str, force_reset: bool = False):
        """
        元音声を設定（新しいセッションを開始）

        Args:
            audio_path: 元音声のパス
            force_reset: Trueの場合、既存の状態を強制リセット
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 既に同じファイルが設定されていて、リセットしない場合はスキップ
        if not force_reset and self.original_audio_path == audio_path:
            return

        # 新しいセッションを開始
        self.original_audio_path = audio_path

        # 元音声のバックアップを作成
        backup_dir = os.path.join(self.state_dir, 'original_audio_backup')
        os.makedirs(backup_dir, exist_ok=True)

        basename = os.path.basename(audio_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{os.path.splitext(basename)[0]}_{timestamp}.wav"
        self.original_audio_backup_path = os.path.join(backup_dir, backup_name)

        shutil.copy2(audio_path, self.original_audio_backup_path)

        # 累積パラメータをリセット
        self.cumulative_params = np.zeros(10)
        self.edit_history = []

        self._save_state()

        print(f"[CumulativeDSP] New session started")
        print(f"  Original audio: {audio_path}")
        print(f"  Backup created: {self.original_audio_backup_path}")

    def continue_session(self, audio_path: str = None) -> bool:
        """
        既存のセッションを継続

        Args:
            audio_path: 確認用のオーディオパス（Noneの場合はチェックしない）

        Returns:
            セッションを継続できたかどうか
        """
        if self.original_audio_backup_path is None:
            return False

        if not os.path.exists(self.original_audio_backup_path):
            print(f"[CumulativeDSP] Warning: Backup file not found, starting new session")
            return False

        if audio_path is not None and audio_path != self.original_audio_path:
            # 異なるファイルが指定された場合は新しいセッションとして扱う
            return False

        print(f"[CumulativeDSP] Continuing existing session")
        print(f"  Original audio: {self.original_audio_path}")
        print(f"  Current cumulative params: {self.get_cumulative_params_summary()}")

        return True

    def get_cumulative_params_summary(self) -> str:
        """累積パラメータのサマリーを取得"""
        summary_parts = []
        for i, name in enumerate(self.PARAM_NAMES):
            if abs(self.cumulative_params[i]) > 0.01:
                summary_parts.append(f"{name}={self.cumulative_params[i]:+.2f}")

        if not summary_parts:
            return "(no changes)"

        return ", ".join(summary_parts)

    def apply_deceleration(self, dsp_diff: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        減速ロジックを適用

        既に極端な値に達しているパラメータは変化量を減少させる
        ΔP_effective = ΔP_model * (1 - |P_total| / P_max)

        Args:
            dsp_diff: モデルが出力したDSPパラメータ差分（10次元）
            verbose: 詳細表示

        Returns:
            減速後のDSPパラメータ差分
        """
        effective_diff = np.zeros_like(dsp_diff)

        if verbose:
            print("\n[Deceleration Logic]")
            print(f"  {'Param':<12} {'Current':>8} {'Model Δ':>8} {'Factor':>8} {'Effective':>8}")
            print("  " + "-" * 48)

        for i, name in enumerate(self.PARAM_NAMES):
            p_current = self.cumulative_params[i]
            p_diff = dsp_diff[i]

            # パラメータの上下限を取得
            p_min, p_max = self.PARAM_LIMITS[name]

            # 変化の方向に応じた限界値を使用
            if p_diff > 0:
                # 増加方向：上限までの余裕
                p_limit = p_max
            else:
                # 減少方向：下限までの余裕
                p_limit = p_min

            # 変化が現在値を0に近づける方向かどうか
            moving_toward_zero = (p_current > 0 and p_diff < 0) or (p_current < 0 and p_diff > 0)

            if moving_toward_zero:
                # 0に戻る方向の変化は減速しない
                deceleration_factor = 1.0
            else:
                # 0から離れる方向の変化は、限界に近づくほど減速
                if abs(p_limit) > 0.01:
                    distance_ratio = abs(p_current) / abs(p_limit)
                    deceleration_factor = max(0.0, 1.0 - distance_ratio)
                else:
                    deceleration_factor = 1.0

            # 限界を超えている場合
            if (p_current >= p_max and p_diff > 0) or (p_current <= p_min and p_diff < 0):
                # さらに限界を超える方向は完全にブロック
                deceleration_factor = 0.0

            effective_diff[i] = p_diff * deceleration_factor

            if verbose and (abs(p_diff) > 0.001 or abs(p_current) > 0.01):
                print(f"  {name:<12} {p_current:>+8.3f} {p_diff:>+8.3f} "
                      f"{deceleration_factor:>8.2f} {effective_diff[i]:>+8.3f}")

        return effective_diff

    def update_parameters(self, dsp_diff: np.ndarray, source_onoma: str = None,
                          target_onoma: str = None):
        """
        累積パラメータを更新

        Args:
            dsp_diff: 今回の編集によるDSPパラメータ差分
            source_onoma: ソースオノマトペ（履歴用）
            target_onoma: ターゲットオノマトペ（履歴用）
        """
        # 累積パラメータを更新
        self.cumulative_params = self.cumulative_params + dsp_diff

        # 上下限でクリップ
        for i, name in enumerate(self.PARAM_NAMES):
            p_min, p_max = self.PARAM_LIMITS[name]
            self.cumulative_params[i] = np.clip(self.cumulative_params[i], p_min, p_max)

        # 履歴に追加
        edit_entry = {
            'timestamp': datetime.now().isoformat(),
            'dsp_diff': dsp_diff.tolist(),
            'cumulative_after': self.cumulative_params.tolist(),
            'source_onomatopoeia': source_onoma,
            'target_onomatopoeia': target_onoma
        }
        self.edit_history.append(edit_entry)

        # 状態を保存
        self._save_state()

    def render(self, output_path: str, verbose: bool = True) -> str:
        """
        元音声に累積パラメータを適用してレンダリング

        Args:
            output_path: 出力ファイルパス
            verbose: 詳細表示

        Returns:
            出力ファイルパス
        """
        if self.original_audio_backup_path is None:
            raise ValueError("No original audio set. Call set_original_audio() first.")

        if not os.path.exists(self.original_audio_backup_path):
            raise FileNotFoundError(f"Backup file not found: {self.original_audio_backup_path}")

        # 累積パラメータをマッピング
        mapped_params = self.mapper.map_parameters(self.cumulative_params)

        if verbose:
            print("\n[Cumulative Render]")
            print(f"  Source: {self.original_audio_backup_path}")
            print(f"  Output: {output_path}")
            print(f"\n  Cumulative Parameters (normalized):")
            for i, name in enumerate(self.PARAM_NAMES):
                if abs(self.cumulative_params[i]) > 0.001:
                    print(f"    {name:<15}: {self.cumulative_params[i]:>+8.4f}")

            print(f"\n  Mapped Parameters:")
            for key, value in mapped_params.items():
                if 'db' in key:
                    print(f"    {key:<25}: {value:>+8.2f} dB")
                elif 'ratio' in key:
                    print(f"    {key:<25}: {value:>8.2f}x")
                else:
                    print(f"    {key:<25}: {value:>+8.2f}")

        # 出力ディレクトリを作成
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 元音声から直接レンダリング
        self.dsp_engine.process_audio_file(
            self.original_audio_backup_path,
            output_path,
            mapped_params
        )

        return output_path

    def get_current_state(self) -> Dict:
        """現在の状態を取得"""
        return {
            'original_audio': self.original_audio_path,
            'backup_audio': self.original_audio_backup_path,
            'cumulative_params': self.cumulative_params.tolist(),
            'cumulative_params_named': {
                name: self.cumulative_params[i]
                for i, name in enumerate(self.PARAM_NAMES)
            },
            'edit_count': len(self.edit_history),
            'param_limits': self.PARAM_LIMITS
        }

    def reset(self):
        """状態をリセット"""
        self.cumulative_params = np.zeros(10)
        self.original_audio_path = None
        self.original_audio_backup_path = None
        self.edit_history = []
        self._save_state()
        print("[CumulativeDSP] State reset")

    def undo_last_edit(self) -> bool:
        """最後の編集を取り消す"""
        if len(self.edit_history) < 2:
            if len(self.edit_history) == 1:
                # 最初の編集を取り消す
                self.cumulative_params = np.zeros(10)
                self.edit_history = []
                self._save_state()
                return True
            return False

        # 最後の編集を削除
        self.edit_history.pop()

        # 一つ前の状態に戻る
        last_state = self.edit_history[-1]
        self.cumulative_params = np.array(last_state['cumulative_after'])

        self._save_state()
        return True

    def get_edit_history(self, limit: int = None) -> list:
        """編集履歴を取得"""
        if limit:
            return self.edit_history[-limit:]
        return self.edit_history


def test_cumulative_manager():
    """累積DSPマネージャーのテスト"""
    print("=" * 70)
    print("Cumulative DSP Manager Test")
    print("=" * 70)

    manager = CumulativeDSPManager()

    # テスト用のダミーDSP差分
    print("\n[Test 1] Initial edit")
    dsp_diff1 = np.array([0.3, 0.1, -0.1, 0.2, -0.2, 0.1, 0.0, 0.2, -0.1, 0.0])
    effective1 = manager.apply_deceleration(dsp_diff1)
    print(f"  Input:  {dsp_diff1}")
    print(f"  Output: {effective1}")

    # シミュレーション：累積パラメータを更新
    manager.cumulative_params = effective1.copy()

    print("\n[Test 2] Second edit (same direction)")
    dsp_diff2 = np.array([0.3, 0.1, -0.1, 0.2, -0.2, 0.1, 0.0, 0.2, -0.1, 0.0])
    effective2 = manager.apply_deceleration(dsp_diff2)
    print(f"  Input:  {dsp_diff2}")
    print(f"  Output: {effective2}")

    # 累積パラメータを更新
    manager.cumulative_params = manager.cumulative_params + effective2

    print("\n[Test 3] Third edit (approaching limits)")
    dsp_diff3 = np.array([0.3, 0.3, -0.2, 0.3, -0.3, 0.3, 0.0, 0.3, -0.2, 0.0])
    effective3 = manager.apply_deceleration(dsp_diff3)
    print(f"  Input:  {dsp_diff3}")
    print(f"  Output: {effective3}")

    print("\n[Test 4] Reverse direction (should not be limited)")
    manager.cumulative_params = np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dsp_diff4 = np.array([-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    effective4 = manager.apply_deceleration(dsp_diff4)
    print(f"  Current gain: {manager.cumulative_params[0]}")
    print(f"  Requested change: {dsp_diff4[0]}")
    print(f"  Effective change: {effective4[0]} (should be close to -0.3)")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == '__main__':
    test_cumulative_manager()
