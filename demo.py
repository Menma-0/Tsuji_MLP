"""
オノマトペMLPモデルのデモスクリプト
"""
import numpy as np
import soundfile as sf
import os
import sys

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.inference import OnomatopoeiaInference
from src.dsp.dsp_engine import DSPEngine
from src.models.mlp_model import DSPParameterMapping


def create_demo_audio(filepath: str, duration: float = 2.0,
                     sample_rate: int = 44100):
    """
    デモ用の音声ファイルを作成

    Args:
        filepath: 保存先ファイルパス
        duration: 長さ（秒）
        sample_rate: サンプリングレート
    """
    print(f"Creating demo audio: {filepath}")

    # 複数の周波数を含む音を生成
    t = np.linspace(0, duration, int(sample_rate * duration))

    # 基音と倍音を重ねる
    audio = (
        0.3 * np.sin(2 * np.pi * 220 * t) +     # A3
        0.2 * np.sin(2 * np.pi * 440 * t) +     # A4
        0.15 * np.sin(2 * np.pi * 880 * t) +    # A5
        0.1 * np.sin(2 * np.pi * 1760 * t)      # A6
    )

    # エンベロープを適用（徐々に減衰）
    envelope = np.exp(-2 * t / duration)
    audio = audio * envelope

    # ノーマライズ
    audio = audio / np.max(np.abs(audio)) * 0.8

    # 保存
    sf.write(filepath, audio, sample_rate)
    print(f"  Duration: {duration}s, Sample rate: {sample_rate}Hz\n")


def demo_inference(onomatopoeia: str, model_path: str, scaler_path: str):
    """
    推論のデモ

    Args:
        onomatopoeia: オノマトペ
        model_path: モデルのパス
        scaler_path: スケーラーのパス
    """
    print("="*60)
    print(f"INFERENCE DEMO: {onomatopoeia}")
    print("="*60 + "\n")

    inference = OnomatopoeiaInference(
        model_path=model_path,
        scaler_path=scaler_path,
        device='cpu'
    )

    result = inference.predict(onomatopoeia)
    inference.print_result(result, verbose=True)


def demo_end_to_end(onomatopoeia: str, input_audio: str, output_audio: str,
                   model_path: str, scaler_path: str):
    """
    エンドツーエンドのデモ

    Args:
        onomatopoeia: オノマトペ
        input_audio: 入力音声ファイル
        output_audio: 出力音声ファイル
        model_path: モデルのパス
        scaler_path: スケーラーのパス
    """
    print("="*60)
    print(f"END-TO-END DEMO: {onomatopoeia}")
    print("="*60 + "\n")

    # 推論
    inference = OnomatopoeiaInference(
        model_path=model_path,
        scaler_path=scaler_path,
        device='cpu'
    )

    result = inference.predict(onomatopoeia)

    print(f"Onomatopoeia: {onomatopoeia}")
    print(f"Input audio: {input_audio}")
    print(f"Output audio: {output_audio}\n")

    # DSPパラメータを表示
    print("Predicted DSP Parameters:")
    for key, value in result['mapped_params'].items():
        if 'db' in key:
            print(f"  {key:<25}: {value:>7.2f} dB")
        elif 'ratio' in key:
            print(f"  {key:<25}: {value:>7.2f}x")
        else:
            print(f"  {key:<25}: {value:>7.2f}")

    # 音声処理
    print(f"\nApplying DSP effects to audio...")
    dsp_engine = DSPEngine(sample_rate=44100)
    dsp_engine.process_audio_file(input_audio, output_audio, result['mapped_params'])

    print("="*60)
    print("DEMO COMPLETED!")
    print("="*60 + "\n")


def main():
    """メイン関数"""
    print("\n" + "="*60)
    print("ONOMATOPOEIA MLP MODEL DEMO")
    print("="*60 + "\n")

    # パスの設定
    model_path = os.path.join('models', 'saved_model.pth')
    scaler_path = os.path.join('models', 'scaler.pkl')

    # モデルの存在確認
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first:")
        print("  cd src")
        print("  python train_with_sample_data.py")
        return

    # デモ音声を作成
    os.makedirs('demo_audio', exist_ok=True)
    input_audio = os.path.join('demo_audio', 'input.wav')
    create_demo_audio(input_audio, duration=2.0)

    # デモ1: 推論のみ
    print("\n" + "="*60)
    print("DEMO 1: INFERENCE ONLY")
    print("="*60 + "\n")

    test_onomatopoeia = ['ガンガン', 'サラサラ', 'キラキラ']

    for onoma in test_onomatopoeia:
        demo_inference(onoma, model_path, scaler_path)

    # デモ2: エンドツーエンド
    print("\n" + "="*60)
    print("DEMO 2: END-TO-END PROCESSING")
    print("="*60 + "\n")

    for onoma in test_onomatopoeia:
        output_audio = os.path.join('demo_audio', f'output_{onoma}.wav')
        demo_end_to_end(onoma, input_audio, output_audio, model_path, scaler_path)

    print("\nDemo audio files created in 'demo_audio/' directory:")
    print(f"  - Input: {input_audio}")
    for onoma in test_onomatopoeia:
        print(f"  - Output ({onoma}): demo_audio/output_{onoma}.wav")

    print("\nYou can listen to the audio files to hear the differences!")


if __name__ == '__main__':
    main()
