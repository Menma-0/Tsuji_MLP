"""
音声にDSPエフェクトを適用するエンジン
"""
import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa
import soundfile as sf
from typing import Optional


class DSPEngine:
    """DSPエフェクトを適用するクラス"""

    def __init__(self, sample_rate: int = 44100):
        """
        Args:
            sample_rate: サンプリングレート
        """
        self.sample_rate = sample_rate

    def apply_gain(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """
        ゲイン（音量）を適用

        Args:
            audio: 音声信号
            gain_db: ゲイン（dB）

        Returns:
            処理後の音声信号
        """
        gain_linear = 10 ** (gain_db / 20.0)
        return audio * gain_linear

    def apply_eq_band(self, audio: np.ndarray, center_freq: float,
                     gain_db: float, q: float = 1.0) -> np.ndarray:
        """
        ピーキングEQを適用

        Args:
            audio: 音声信号
            center_freq: 中心周波数（Hz）
            gain_db: ゲイン（dB）
            q: Q値（帯域幅）

        Returns:
            処理後の音声信号
        """
        if abs(gain_db) < 0.01:
            return audio

        # ピーキングフィルタの設計
        w0 = 2 * np.pi * center_freq / self.sample_rate
        A = 10 ** (gain_db / 40.0)
        alpha = np.sin(w0) / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0

        # フィルタを適用
        filtered = signal.lfilter(b, a, audio)

        return filtered

    def apply_compression(self, audio: np.ndarray, compression_amount: float,
                         threshold: float = -20.0, ratio: float = 4.0) -> np.ndarray:
        """
        シンプルなコンプレッサーを適用

        Args:
            audio: 音声信号
            compression_amount: 圧縮量（-1〜+1）
            threshold: スレッショルド（dB）
            ratio: 圧縮比

        Returns:
            処理後の音声信号
        """
        if abs(compression_amount) < 0.01:
            return audio

        # compression_amountに応じてスレッショルドとレシオを調整
        adjusted_threshold = threshold + (compression_amount * 10)
        adjusted_ratio = 1.0 + (compression_amount + 1) * (ratio - 1) / 2

        # dBに変換
        audio_db = 20 * np.log10(np.abs(audio) + 1e-10)

        # コンプレッション
        mask = audio_db > adjusted_threshold
        compressed_db = audio_db.copy()
        compressed_db[mask] = adjusted_threshold + (audio_db[mask] - adjusted_threshold) / adjusted_ratio

        # リニアに戻す
        compressed_linear = 10 ** (compressed_db / 20.0) * np.sign(audio)

        return compressed_linear

    def apply_transient_shaper(self, audio: np.ndarray, attack: float,
                              sustain: float) -> np.ndarray:
        """
        トランジェントシェイパー（簡易版）

        Args:
            audio: 音声信号
            attack: アタック調整（-1〜+1）
            sustain: サスティン調整（-1〜+1）

        Returns:
            処理後の音声信号
        """
        if abs(attack) < 0.01 and abs(sustain) < 0.01:
            return audio

        # エンベロープを計算
        envelope = np.abs(audio)
        envelope_smooth = signal.lfilter([1-0.99], [1, -0.99], envelope)

        # アタック部分を検出
        diff = np.diff(envelope_smooth, prepend=envelope_smooth[0])
        attack_mask = diff > 0

        # トランジェントを調整
        result = audio.copy()

        # アタック部分を強調/減衰
        if attack > 0:
            result[attack_mask] *= (1 + attack * 0.5)
        else:
            result[attack_mask] *= (1 + attack * 0.3)

        # サスティン部分を調整
        sustain_mask = ~attack_mask
        if sustain > 0:
            result[sustain_mask] *= (1 + sustain * 0.3)
        else:
            result[sustain_mask] *= (1 + sustain * 0.5)

        return result

    def apply_time_stretch(self, audio: np.ndarray, stretch_ratio: float) -> np.ndarray:
        """
        タイムストレッチを適用

        Args:
            audio: 音声信号
            stretch_ratio: 伸縮比率（0.5〜2.0）

        Returns:
            処理後の音声信号
        """
        if abs(stretch_ratio - 1.0) < 0.01:
            return audio

        # librosaを使ってタイムストレッチ
        stretched = librosa.effects.time_stretch(audio, rate=stretch_ratio)

        return stretched

    def apply_all_effects(self, audio: np.ndarray, params: dict) -> np.ndarray:
        """
        すべてのDSPエフェクトを適用

        Args:
            audio: 音声信号
            params: DSPパラメータの辞書

        Returns:
            処理後の音声信号
        """
        # パラメータを取得
        gain_db = params.get('gain_db', 0.0)
        compression = params.get('compression', 0.0)
        eq_sub_db = params.get('eq_sub_db', 0.0)
        eq_low_db = params.get('eq_low_db', 0.0)
        eq_mid_db = params.get('eq_mid_db', 0.0)
        eq_high_db = params.get('eq_high_db', 0.0)
        eq_presence_db = params.get('eq_presence_db', 0.0)
        transient_attack = params.get('transient_attack', 0.0)
        transient_sustain = params.get('transient_sustain', 0.0)
        time_stretch_ratio = params.get('time_stretch_ratio', 1.0)

        processed = audio.copy()

        # 1. EQを適用
        print("Applying EQ...")
        processed = self.apply_eq_band(processed, 80, eq_sub_db, q=1.0)
        processed = self.apply_eq_band(processed, 250, eq_low_db, q=1.0)
        processed = self.apply_eq_band(processed, 1000, eq_mid_db, q=1.0)
        processed = self.apply_eq_band(processed, 4000, eq_high_db, q=1.0)
        processed = self.apply_eq_band(processed, 10000, eq_presence_db, q=1.0)

        # 2. コンプレッションを適用
        print("Applying compression...")
        processed = self.apply_compression(processed, compression)

        # 3. トランジェントシェイパーを適用
        print("Applying transient shaper...")
        processed = self.apply_transient_shaper(processed, transient_attack, transient_sustain)

        # 4. タイムストレッチを適用
        print("Applying time stretch...")
        processed = self.apply_time_stretch(processed, time_stretch_ratio)

        # 5. ゲインを適用
        print("Applying gain...")
        processed = self.apply_gain(processed, gain_db)

        # クリッピング防止
        max_val = np.max(np.abs(processed))
        if max_val > 1.0:
            processed = processed / max_val * 0.95

        return processed

    def process_audio_file(self, input_path: str, output_path: str,
                          params: dict) -> None:
        """
        音声ファイルを処理

        Args:
            input_path: 入力ファイルパス
            output_path: 出力ファイルパス
            params: DSPパラメータの辞書
        """
        print(f"\nProcessing: {input_path}")

        # 音声ファイルを読み込み
        audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True)

        # エフェクトを適用
        processed = self.apply_all_effects(audio, params)

        # 保存
        sf.write(output_path, processed, sr)
        print(f"Saved to: {output_path}\n")


def test_dsp_engine():
    """DSPエンジンのテスト"""
    print("=== DSP Engine Test ===\n")

    # ダミーの音声信号を作成（1秒間のサイン波）
    sample_rate = 44100
    duration = 1.0
    frequency = 440.0  # A4

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.5

    # DSPエンジンを作成
    engine = DSPEngine(sample_rate=sample_rate)

    # テストパラメータ
    test_params = {
        'gain_db': 6.0,
        'compression': 0.5,
        'eq_sub_db': -3.0,
        'eq_low_db': 0.0,
        'eq_mid_db': 3.0,
        'eq_high_db': 6.0,
        'eq_presence_db': -3.0,
        'transient_attack': 0.5,
        'transient_sustain': -0.3,
        'time_stretch_ratio': 1.2
    }

    print("Test parameters:")
    for key, value in test_params.items():
        print(f"  {key}: {value}")

    # エフェクトを適用
    print("\nApplying effects...")
    processed = engine.apply_all_effects(audio, test_params)

    print(f"Original audio shape: {audio.shape}")
    print(f"Processed audio shape: {processed.shape}")
    print(f"Original max amplitude: {np.max(np.abs(audio)):.3f}")
    print(f"Processed max amplitude: {np.max(np.abs(processed)):.3f}")

    print("\nDSP Engine test completed!")


if __name__ == '__main__':
    test_dsp_engine()
