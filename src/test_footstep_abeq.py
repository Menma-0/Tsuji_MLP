"""
足跡音声データに対する編集テスト（累積DSP方針A&B使用）
結果をAB&EQフォルダに保存
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from src.process_with_cumulative_dsp import CumulativeDSPProcessor


def main():
    print("=" * 70)
    print("足跡音声編集テスト（方針A&B）- AB&EQ")
    print("=" * 70)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # パス設定
    model_path = os.path.join(project_root, 'models', 'pair_model.pth')
    scaler_path = os.path.join(project_root, 'models', 'pair_scaler.pkl')
    input_audio = os.path.join(project_root, 'demo_audio', 'test_walk.wav')
    output_dir = os.path.join(project_root, 'AB&EQ')

    os.makedirs(output_dir, exist_ok=True)

    # プロセッサを初期化
    processor = CumulativeDSPProcessor(
        model_path=model_path,
        scaler_path=scaler_path
    )

    # 足跡音の編集ケース
    source_onomatopoeia = 'ジャッジャッ'

    edit_cases = [
        ('タッタッ', 'walk_tatta'),           # 軽い足音
        ('ドスドス', 'walk_dosudosu'),        # 重い足音
        ('サクサク', 'walk_sakusaku'),        # 雪や落ち葉
        ('ペタペタ', 'walk_petapeta'),        # 裸足
        ('コツコツ', 'walk_kotsukotsu'),      # ハイヒール
        ('ドンドン', 'walk_dondon'),          # 力強い足音
        ('パタパタ', 'walk_patapata'),        # スリッパ
        ('ザクザク', 'walk_zakuzaku'),        # 砂利や雪を踏む
        ('トコトコ', 'walk_tokotoko'),        # 小刻みな足音
        ('ドタドタ', 'walk_dotadota'),        # 慌ただしい足音
    ]

    print(f"\n入力音声: {input_audio}")
    print(f"出力先: {output_dir}")
    print(f"元のオノマトペ: {source_onomatopoeia}")
    print(f"\n編集ケース数: {len(edit_cases)}")

    # 元音声もコピー
    original_copy = os.path.join(output_dir, 'walk_original.wav')
    shutil.copy2(input_audio, original_copy)
    print(f"\n元音声コピー: {original_copy}")

    results = []

    for target, output_name in edit_cases:
        print(f"\n{'='*70}")
        print(f"編集: {source_onomatopoeia} → {target}")
        print(f"{'='*70}")

        # 各編集ごとにセッションを新規開始
        processor.start_session(input_audio)

        output_path = os.path.join(output_dir, f'{output_name}.wav')

        try:
            result = processor.process(
                source_onomatopoeia=source_onomatopoeia,
                target_onomatopoeia=target,
                output_audio_path=output_path,
                verbose=True
            )
            results.append({
                'target': target,
                'output': output_name,
                'params': result['cumulative_params'],
                'success': True
            })
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'target': target,
                'output': output_name,
                'success': False,
                'error': str(e)
            })

    # サマリー
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n処理完了: {sum(1 for r in results if r['success'])}/{len(results)}")
    print(f"出力フォルダ: {output_dir}")

    print("\n生成ファイル:")
    print(f"  - walk_original.wav (元音声)")
    for result in results:
        if result['success']:
            print(f"  - {result['output']}.wav ({source_onomatopoeia} → {result['target']})")

    # パラメータ比較表
    print("\n\nDSPパラメータ比較:")
    param_names = ['gain', 'comp', 'sub', 'low', 'mid', 'high', 'pres', 'atk', 'sus', 'str']

    header = f"{'Target':<12}"
    for name in param_names[:7]:
        header += f"{name:>8}"
    print(header)
    print("-" * 68)

    for result in results:
        if result['success']:
            row = f"{result['target']:<12}"
            for i in range(7):
                row += f"{result['params'][i]:>+8.3f}"
            print(row)


if __name__ == '__main__':
    main()
