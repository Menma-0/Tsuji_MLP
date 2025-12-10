"""
様々な音声に対して編集操作をテスト
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from src.process_with_pair_model import PairModelProcessor

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # モデルとスケーラーのパス
    model_path = os.path.join(project_root, 'models', 'pair_model.pth')
    scaler_path = os.path.join(project_root, 'models', 'pair_scaler.pkl')

    # 出力ディレクトリ
    output_dir = os.path.join(project_root, 'demo_audio', 'editing_test')
    os.makedirs(output_dir, exist_ok=True)

    # プロセッサを初期化
    processor = PairModelProcessor(
        model_path=model_path,
        scaler_path=scaler_path
    )

    # テストケース: (入力音声, ソースオノマトペ, ターゲットオノマトペ, 出力名)
    test_cases = [
        # ベル音
        ('demo_audio/diverse_test/01_bell_original.wav', 'チリン', 'ガンガン', 'bell_chirin_to_gangan.wav'),
        ('demo_audio/diverse_test/01_bell_original.wav', 'チリン', 'ドスン', 'bell_chirin_to_dosun.wav'),
        ('demo_audio/diverse_test/01_bell_original.wav', 'チリン', 'キラキラ', 'bell_chirin_to_kirakira.wav'),

        # コイン音
        ('demo_audio/diverse_test/02_coin_original.wav', 'チャリン', 'ガチャン', 'coin_charin_to_gachan.wav'),
        ('demo_audio/diverse_test/02_coin_original.wav', 'チャリン', 'ドン', 'coin_charin_to_don.wav'),
        ('demo_audio/diverse_test/02_coin_original.wav', 'チャリン', 'カラカラ', 'coin_charin_to_karakara.wav'),

        # 木の音
        ('demo_audio/diverse_test/03_wood_original.wav', 'コツ', 'ドン', 'wood_kotsu_to_don.wav'),
        ('demo_audio/diverse_test/03_wood_original.wav', 'コツ', 'ガツン', 'wood_kotsu_to_gatsun.wav'),
        ('demo_audio/diverse_test/03_wood_original.wav', 'コツ', 'パキッ', 'wood_kotsu_to_paki.wav'),
    ]

    print("=" * 70)
    print("編集操作テスト")
    print("=" * 70)

    results = []

    for input_audio, source, target, output_name in test_cases:
        input_path = os.path.join(project_root, input_audio)
        output_path = os.path.join(output_dir, output_name)

        if not os.path.exists(input_path):
            print(f"\n[SKIP] 入力ファイルが見つかりません: {input_path}")
            continue

        print(f"\n\n{'='*70}")
        print(f"TEST: {source} → {target}")
        print(f"Input: {input_audio}")
        print(f"{'='*70}")

        try:
            result = processor.process(
                source_onomatopoeia=source,
                target_onomatopoeia=target,
                input_audio_path=input_path,
                output_audio_path=output_path,
                verbose=True
            )
            results.append({
                'input': input_audio,
                'source': source,
                'target': target,
                'output': output_name,
                'params': result['mapped_params']
            })
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # サマリー
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n処理完了: {len(results)} ファイル")
    print(f"出力先: {output_dir}")

    print(f"\n{'Case':<30} {'gain':>8} {'eq_low':>8} {'eq_mid':>8} {'eq_high':>8} {'attack':>8}")
    print("-" * 80)

    for r in results:
        case = f"{r['source']}→{r['target']}"
        p = r['params']
        print(f"{case:<30} {p.get('gain_db', 0):>+8.2f} {p.get('eq_low_db', 0):>+8.2f} "
              f"{p.get('eq_mid_db', 0):>+8.2f} {p.get('eq_high_db', 0):>+8.2f} "
              f"{p.get('transient_attack', 0):>+8.2f}")


if __name__ == '__main__':
    main()
