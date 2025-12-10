"""
累積DSPパラメータ管理

方針A: 元音+累積パラメータから毎回再レンダリング
方針B: パラメータ上限・下限と減速ロジック
"""
import numpy as np
from typing import Dict, Optional
import os
import shutil
from datetime import datetime


class CumulativeDSPManager:
    """
    累積DSPパラメータを管理するクラス

    - 元の音声を保持し、累積パラメータから毎回再レンダリング（方針A）
    - パラメータに上限・下限を設定し、減速ロジックを適用（方針B）
    """

    # DSPパラメータの上限・下限（正規化された値: -1.0 〜 +1.0）
    PARAM_LIMITS = {
        'gain': (-0.5, 0.5),           # ±12dB相当
        'compression': (-1.0, 1.0),     # フル範囲
        'eq_sub': (-0.5, 0.5),          # ±12dB相当
        'eq_low': (-0.5, 0.5),          # ±12dB相当
        'eq_mid': (-0.6, 0.6),          # ±14.4dB相当
        'eq_high': (-0.5, 0.5),         # ±12dB相当
        'eq_presence': (-0.5, 0.5),     # ±12dB相当
        'attack': (-0.8, 0.8),          # トランジェント調整
        'sustain': (-0.8, 0.8),         # サスティン調整
        'time_stretch': (-0.5, 0.5),    # 0.5x 〜 2.0x
    }

    PARAM_NAMES = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                   'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']

    def __init__(self, backup_dir: str = 'history/original_audio_backup'):
        """
        Args:
            backup_dir: 元音声のバックアップディレクトリ
        """
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

        # 累積パラメータ（10次元、初期値0）
        self.cumulative_params = np.zeros(10)

        # 元音声のパス（セッション開始時に設定）
        self.original_audio_path: Optional[str] = None
        self.backup_audio_path: Optional[str] = None

        # 編集回数
        self.edit_count = 0

    def start_session(self, original_audio_path: str) -> str:
        """
        新しい編集セッションを開始

        Args:
            original_audio_path: 元音声のパス

        Returns:
            バックアップされた元音声のパス
        """
        self.original_audio_path = original_audio_path
        self.cumulative_params = np.zeros(10)
        self.edit_count = 0

        # 元音声をバックアップ
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.basename(original_audio_path)
        name, ext = os.path.splitext(filename)
        backup_filename = f"{name}_{timestamp}{ext}"
        self.backup_audio_path = os.path.join(self.backup_dir, backup_filename)

        shutil.copy2(original_audio_path, self.backup_audio_path)

        return self.backup_audio_path

    def apply_delta_with_deceleration(
        self,
        delta_params: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
        """
        差分パラメータを累積に適用（減速ロジック付き）

        方針B: 現在の値に応じてΔを減速させる
        ΔP_effective = ΔP_model * (1 - |P_current| / P_max)

        Args:
            delta_params: モデルが出力した差分パラメータ（10次元）
            verbose: 詳細表示

        Returns:
            実際に適用された差分（減速後）
        """
        effective_delta = np.zeros(10)

        if verbose:
            print("\n[Deceleration Logic]")
            print(f"  {'Param':<13} {'Current':>8} {'Model Δ':>8} {'Factor':>8} {'Effective':>8}")
            print("  " + "-" * 48)

        for i, name in enumerate(self.PARAM_NAMES):
            p_min, p_max = self.PARAM_LIMITS[name]
            p_range = p_max - p_min
            p_current = self.cumulative_params[i]
            delta = delta_params[i]

            # 現在値が上限/下限に近いほど、その方向への変化を抑制
            if delta > 0:
                # 増加方向: 上限までの余裕に応じて減速
                headroom = (p_max - p_current) / (p_range / 2)
                headroom = np.clip(headroom, 0, 1)
                factor = headroom
            else:
                # 減少方向: 下限までの余裕に応じて減速
                headroom = (p_current - p_min) / (p_range / 2)
                headroom = np.clip(headroom, 0, 1)
                factor = headroom

            effective_delta[i] = delta * factor

            if verbose:
                print(f"  {name:<13} {p_current:>+8.3f} {delta:>+8.3f} {factor:>8.2f} {effective_delta[i]:>+8.3f}")

        # 累積パラメータを更新
        self.cumulative_params += effective_delta

        # 上限・下限でクリップ
        for i, name in enumerate(self.PARAM_NAMES):
            p_min, p_max = self.PARAM_LIMITS[name]
            self.cumulative_params[i] = np.clip(self.cumulative_params[i], p_min, p_max)

        self.edit_count += 1

        if verbose:
            print(f"\n  Cumulative params after update: ", end='')
            parts = []
            for i, name in enumerate(self.PARAM_NAMES):
                if abs(self.cumulative_params[i]) > 0.001:
                    parts.append(f"{name}={self.cumulative_params[i]:+.2f}")
            print(', '.join(parts) if parts else '(all zero)')

        return effective_delta

    def get_cumulative_params(self) -> np.ndarray:
        """現在の累積パラメータを取得"""
        return self.cumulative_params.copy()

    def get_cumulative_params_dict(self) -> Dict[str, float]:
        """累積パラメータを辞書形式で取得"""
        return {name: self.cumulative_params[i] for i, name in enumerate(self.PARAM_NAMES)}

    def get_original_audio_path(self) -> Optional[str]:
        """バックアップされた元音声のパスを取得"""
        return self.backup_audio_path

    def reset(self):
        """セッションをリセット"""
        self.cumulative_params = np.zeros(10)
        self.original_audio_path = None
        self.backup_audio_path = None
        self.edit_count = 0

    def get_status(self) -> dict:
        """現在の状態を取得"""
        return {
            'edit_count': self.edit_count,
            'original_audio': self.original_audio_path,
            'backup_audio': self.backup_audio_path,
            'cumulative_params': self.get_cumulative_params_dict()
        }


# グローバルインスタンス（セッション管理用）
_global_manager: Optional[CumulativeDSPManager] = None


def get_cumulative_manager() -> CumulativeDSPManager:
    """グローバルな累積DSPマネージャーを取得"""
    global _global_manager
    if _global_manager is None:
        _global_manager = CumulativeDSPManager()
    return _global_manager


def reset_cumulative_manager():
    """グローバルマネージャーをリセット"""
    global _global_manager
    if _global_manager is not None:
        _global_manager.reset()
