"""
サンプルデータセットを使ってモデルを学習するスクリプト
"""
import pandas as pd
import torch
import pickle
import os
import sys

sys.path.append(os.path.dirname(__file__))

from data.data_loader import OnomatopoeiaDataset
from train import train_pytorch_model, train_sklearn_model, Evaluator


def main():
    """メイン関数"""
    print("\n" + "="*60)
    print("TRAINING WITH SAMPLE DATASET")
    print("="*60 + "\n")

    # サンプルデータセットを読み込み
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_dataset.csv')

    if not os.path.exists(data_path):
        print(f"Error: Sample dataset not found at {data_path}")
        print("Please run: python src/utils/dataset_creator.py")
        return

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # データセットをロード
    dataset = OnomatopoeiaDataset()
    dataset.load_data(df)

    # スコアでフィルタリング
    dataset.filter_by_score(confidence_threshold=4.0, acceptance_threshold=4.0)

    # データ分割
    dataset.split_data(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    # PyTorchモデルを学習
    print("\n" + "="*60)
    print("TRAINING PYTORCH MODEL")
    print("="*60)

    pytorch_model, trainer, pytorch_results, dataset = train_pytorch_model(
        dataset, n_epochs=150, batch_size=32, learning_rate=1e-3
    )

    # scikit-learnモデルを学習（比較用）
    print("\n" + "="*60)
    print("TRAINING SCIKIT-LEARN MODEL")
    print("="*60)

    sklearn_model, sklearn_results = train_sklearn_model(dataset, max_iter=1000)

    # 結果の比較
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"\nPyTorch Model:")
    print(f"  Overall R2: {pytorch_results['overall_r2']:.4f}")
    print(f"  Overall MSE: {pytorch_results['overall_mse']:.4f}")

    print(f"\nscikit-learn Model:")
    print(f"  Overall R2: {sklearn_results['overall_r2']:.4f}")
    print(f"  Overall MSE: {sklearn_results['overall_mse']:.4f}")

    print("\n" + "="*60 + "\n")

    # モデルとスケーラーを保存
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, 'saved_model.pth')
    scaler_path = os.path.join(save_dir, 'scaler.pkl')

    # PyTorchモデルを保存
    torch.save(pytorch_model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # スケーラーを保存
    with open(scaler_path, 'wb') as f:
        pickle.dump(dataset.scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    print("\nYou can now use the following commands:")
    print(f"  1. Inference: python src/inference.py -o ガンガン")
    print(f"  2. End-to-end: python src/onoma2dsp.py -o ガンガン -i input.wav -p output.wav")


if __name__ == '__main__':
    main()
