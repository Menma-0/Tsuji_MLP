"""
Pair Model CLI Tool
差分モデル（ペアモデル）用の対話型コマンドラインインターフェース
"""
import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

# プロジェクトルートを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from src.process_with_pair_model import PairModelProcessor


class PairHistoryManager:
    """ペアモデル用の編集履歴を管理するクラス"""

    def __init__(self, history_file='history/pair_edit_history.json'):
        self.history_file = history_file
        self.history_dir = os.path.dirname(history_file)

        # 履歴ディレクトリを作成
        if self.history_dir:
            os.makedirs(self.history_dir, exist_ok=True)

        # 履歴ファイルが存在しなければ初期化
        if not os.path.exists(self.history_file):
            self._save_history([])

    def _load_history(self):
        """履歴を読み込む"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

    def _save_history(self, history):
        """履歴を保存"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def add_entry(self, input_audio, source_onoma, target_onoma, output_audio,
                  feature_diff_magnitude, dsp_diff_raw, mapped_params):
        """履歴エントリを追加"""
        history = self._load_history()

        entry = {
            'id': len(history) + 1,
            'timestamp': datetime.now().isoformat(),
            'input_audio': input_audio,
            'source_onomatopoeia': source_onoma,
            'target_onomatopoeia': target_onoma,
            'output_audio': output_audio,
            'feature_diff_magnitude': feature_diff_magnitude,
            'dsp_diff_raw': dsp_diff_raw,
            'mapped_params': mapped_params
        }

        history.append(entry)
        self._save_history(history)

        return entry['id']

    def get_history(self, limit=None):
        """履歴を取得"""
        history = self._load_history()
        if limit:
            return history[-limit:]
        return history

    def search_history(self, query):
        """履歴を検索"""
        history = self._load_history()
        results = []

        for entry in history:
            if (query.lower() in entry['source_onomatopoeia'].lower() or
                query.lower() in entry['target_onomatopoeia'].lower() or
                query.lower() in entry['input_audio'].lower()):
                results.append(entry)

        return results

    def get_entry(self, entry_id):
        """特定のエントリを取得"""
        history = self._load_history()
        for entry in history:
            if entry['id'] == entry_id:
                return entry
        return None

    def clear_history(self):
        """履歴をクリア"""
        self._save_history([])

    def replay_entry(self, entry_id, processor, new_output=None):
        """履歴エントリを再実行"""
        entry = self.get_entry(entry_id)
        if entry is None:
            return None

        input_audio = entry['input_audio']
        source_onoma = entry['source_onomatopoeia']
        target_onoma = entry['target_onomatopoeia']

        if new_output is None:
            # 新しい出力ファイル名を生成
            input_path = Path(input_audio)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_output = f"output/{input_path.stem}_{source_onoma}_to_{target_onoma}_replay_{timestamp}.wav"

        return processor.process(
            source_onomatopoeia=source_onoma,
            target_onomatopoeia=target_onoma,
            input_audio_path=input_audio,
            output_audio_path=new_output,
            verbose=True
        )


class PairModelCLI:
    """ペアモデルのCLIインターフェース"""

    def __init__(self):
        self.processor = None
        self.history_manager = PairHistoryManager()
        self.current_settings = {
            'model_path': 'models/pair_model.pth',
            'scaler_path': 'models/pair_scaler.pkl',
            'sample_rate': 44100,
            'lambda_att': 0.5,  # Attention補正の強度
            'cumulative_mode': True  # 累積モード（デフォルトON）
        }

    def _initialize_processor(self):
        """プロセッサを初期化（遅延初期化）"""
        if self.processor is None:
            print("Loading pair model...")
            self.processor = PairModelProcessor(
                model_path=self.current_settings['model_path'],
                scaler_path=self.current_settings['scaler_path'],
                sample_rate=self.current_settings['sample_rate'],
                lambda_att=self.current_settings['lambda_att'],
                use_cumulative=self.current_settings['cumulative_mode']
            )
            mode_str = "ON" if self.current_settings['cumulative_mode'] else "OFF"
            print(f"Pair model loaded. (lambda_att={self.current_settings['lambda_att']}, cumulative={mode_str})\n")

    def process_audio(self, input_audio, source_onoma, target_onoma, output_audio=None):
        """音声を処理"""
        self._initialize_processor()

        # 出力パスが指定されていなければ自動生成
        if output_audio is None:
            input_path = Path(input_audio)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_audio = f"output/{input_path.stem}_{source_onoma}_to_{target_onoma}_{timestamp}.wav"

        # 出力ディレクトリを作成
        output_dir = os.path.dirname(output_audio)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"\nProcessing (Pair Model):")
        print(f"  Input:  {input_audio}")
        print(f"  Source: {source_onoma}")
        print(f"  Target: {target_onoma}")
        print(f"  Output: {output_audio}")
        print()

        # 処理実行
        result = self.processor.process(
            source_onomatopoeia=source_onoma,
            target_onomatopoeia=target_onoma,
            input_audio_path=input_audio,
            output_audio_path=output_audio,
            verbose=True
        )

        # 履歴に記録
        entry_id = self.history_manager.add_entry(
            input_audio=input_audio,
            source_onoma=source_onoma,
            target_onoma=target_onoma,
            output_audio=output_audio,
            feature_diff_magnitude=result['feature_diff_magnitude'],
            dsp_diff_raw=result['dsp_diff_raw'],
            mapped_params=result['mapped_params']
        )

        print(f"\n[Success] Saved to: {output_audio}")
        print(f"[History] Entry #{entry_id} recorded")

        # 累積モードの情報を表示
        if result.get('cumulative_mode') and 'cumulative' in result:
            cumul = result['cumulative']
            print(f"[Cumulative] Edit #{cumul['edit_number']} (continuing: {cumul['is_continuing']})")

        return result

    def show_cumulative_state(self):
        """累積状態を表示"""
        self._initialize_processor()

        if not self.processor.use_cumulative:
            print("Cumulative mode is disabled.")
            return

        state = self.processor.get_cumulative_state()
        if state is None:
            print("No cumulative state available.")
            return

        print(f"\n{'='*60}")
        print("Cumulative DSP State")
        print(f"{'='*60}\n")

        print(f"Original audio: {state.get('original_audio', 'Not set')}")
        print(f"Total edits: {state.get('edit_count', 0)}")

        print("\nCumulative Parameters:")
        param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                      'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']
        params = state.get('cumulative_params', [0] * 10)
        limits = state.get('param_limits', {})

        print(f"  {'Param':<15} {'Current':>10} {'Min':>10} {'Max':>10}")
        print("  " + "-" * 45)
        for i, name in enumerate(param_names):
            p_min, p_max = limits.get(name, (-1, 1))
            print(f"  {name:<15} {params[i]:>+10.4f} {p_min:>+10.2f} {p_max:>+10.2f}")

    def reset_cumulative(self):
        """累積状態をリセット"""
        self._initialize_processor()

        if not self.processor.use_cumulative:
            print("Cumulative mode is disabled.")
            return

        self.processor.reset_cumulative()
        print("Cumulative state has been reset.")

    def undo_last_edit(self):
        """最後の編集を取り消す"""
        self._initialize_processor()

        if not self.processor.use_cumulative:
            print("Cumulative mode is disabled. Undo is not available.")
            return False

        if self.processor.undo_last_edit():
            print("Last edit undone successfully.")
            return True
        else:
            print("No edits to undo.")
            return False

    def new_session(self, input_audio):
        """新しい編集セッションを開始"""
        self._initialize_processor()

        if not self.processor.use_cumulative:
            print("Cumulative mode is disabled.")
            return

        if not os.path.exists(input_audio):
            print(f"Error: File not found: {input_audio}")
            return

        self.processor.start_new_session(input_audio)
        print(f"New session started with: {input_audio}")

    def show_history(self, limit=10):
        """履歴を表示"""
        history = self.history_manager.get_history(limit=limit)

        if not history:
            print("No history found.")
            return

        print(f"\n{'='*80}")
        print(f"Edit History - Pair Model (Last {len(history)} entries)")
        print(f"{'='*80}\n")

        for entry in history:
            print(f"[#{entry['id']}] {entry['timestamp']}")
            print(f"  {entry['source_onomatopoeia']} -> {entry['target_onomatopoeia']}")
            print(f"  Input:  {entry['input_audio']}")
            print(f"  Output: {entry['output_audio']}")
            print(f"  Feature Diff: {entry['feature_diff_magnitude']:.4f}")
            print()

    def show_history_detail(self, entry_id):
        """履歴の詳細を表示"""
        entry = self.history_manager.get_entry(entry_id)

        if entry is None:
            print(f"Entry #{entry_id} not found.")
            return

        print(f"\n{'='*80}")
        print(f"History Entry #{entry['id']} - Detail")
        print(f"{'='*80}\n")

        print(f"Timestamp: {entry['timestamp']}")
        print(f"Source:    {entry['source_onomatopoeia']}")
        print(f"Target:    {entry['target_onomatopoeia']}")
        print(f"Input:     {entry['input_audio']}")
        print(f"Output:    {entry['output_audio']}")
        print(f"Feature Diff Magnitude: {entry['feature_diff_magnitude']:.4f}")

        print("\nRaw DSP Differences:")
        param_names = ['gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
                      'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch']
        for i, name in enumerate(param_names):
            print(f"  {name:<15}: {entry['dsp_diff_raw'][i]:>+8.4f}")

        print("\nMapped DSP Parameters:")
        for key, value in entry['mapped_params'].items():
            if 'db' in key:
                print(f"  {key:<25}: {value:>+8.2f} dB")
            elif 'ratio' in key:
                print(f"  {key:<25}: {value:>8.2f}x")
            else:
                print(f"  {key:<25}: {value:>+8.2f}")

    def search_history(self, query):
        """履歴を検索"""
        results = self.history_manager.search_history(query)

        if not results:
            print(f"No results found for: {query}")
            return

        print(f"\n{'='*80}")
        print(f"Search Results for '{query}' ({len(results)} entries)")
        print(f"{'='*80}\n")

        for entry in results:
            print(f"[#{entry['id']}] {entry['timestamp']}")
            print(f"  {entry['source_onomatopoeia']} -> {entry['target_onomatopoeia']}")
            print(f"  Input:  {entry['input_audio']}")
            print(f"  Output: {entry['output_audio']}")
            print()

    def replay_history(self, entry_id, new_output=None):
        """履歴エントリを再実行"""
        self._initialize_processor()

        entry = self.history_manager.get_entry(entry_id)
        if entry is None:
            print(f"Entry #{entry_id} not found.")
            return None

        print(f"\nReplaying Entry #{entry_id}:")
        print(f"  Source: {entry['source_onomatopoeia']}")
        print(f"  Target: {entry['target_onomatopoeia']}")

        result = self.history_manager.replay_entry(
            entry_id,
            self.processor,
            new_output
        )

        if result:
            # 新しい履歴として記録
            new_entry_id = self.history_manager.add_entry(
                input_audio=entry['input_audio'],
                source_onoma=entry['source_onomatopoeia'],
                target_onoma=entry['target_onomatopoeia'],
                output_audio=result['output_audio'],
                feature_diff_magnitude=result['feature_diff_magnitude'],
                dsp_diff_raw=result['dsp_diff_raw'],
                mapped_params=result['mapped_params']
            )
            print(f"\n[Replay Success] New entry #{new_entry_id} created")

        return result

    def show_settings(self):
        """現在の設定を表示"""
        print("\nCurrent Settings:")
        print(f"  Model:       {self.current_settings['model_path']}")
        print(f"  Scaler:      {self.current_settings['scaler_path']}")
        print(f"  Sample Rate: {self.current_settings['sample_rate']}")
        print(f"  Lambda Att:  {self.current_settings['lambda_att']} (Attention strength)")
        mode_str = "ON" if self.current_settings['cumulative_mode'] else "OFF"
        print(f"  Cumulative:  {mode_str} (multi-edit quality preservation)")
        print()

    def update_settings(self, **kwargs):
        """設定を更新"""
        for key, value in kwargs.items():
            if key in self.current_settings:
                self.current_settings[key] = value
                print(f"Updated {key}: {value}")

        # プロセッサをリセット（次回使用時に再初期化）
        if any(k in kwargs for k in ['model_path', 'scaler_path', 'sample_rate', 'lambda_att']):
            self.processor = None

    def clear_history(self):
        """履歴をクリア"""
        self.history_manager.clear_history()
        print("History cleared.")

    def interactive_mode(self):
        """対話モード"""
        print("="*80)
        print(" "*15 + "Pair Model CLI - Interactive Mode")
        print("="*80)
        print("\nCommands:")
        print("  process <input> <source> <target> [output] - Process audio")
        print("  set <param> <value>                         - Update settings")
        print("  settings                                    - Show current settings")
        print("  history [limit]                             - Show edit history")
        print("  detail <id>                                 - Show history detail")
        print("  search <query>                              - Search history")
        print("  replay <id> [output]                        - Replay history entry")
        print("  clear                                       - Clear history")
        print("  help                                        - Show this help")
        print("  quit / exit                                 - Exit")
        print("\nCumulative Mode Commands:")
        print("  state                                       - Show cumulative DSP state")
        print("  newsession <input>                          - Start new editing session")
        print("  undo                                        - Undo last edit")
        print("  resetcumul                                  - Reset cumulative params")
        print("\nExamples:")
        print("  > process demo_audio/test_walk.wav jiajia tatta")
        print("  > history 5")
        print("  > state")
        print("  > undo")
        print("="*80)

        while True:
            try:
                command = input("\npair> ").strip()

                if not command:
                    continue

                parts = command.split()
                cmd = parts[0].lower()

                if cmd in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                elif cmd == 'help':
                    self.interactive_mode()
                    return

                elif cmd == 'process':
                    if len(parts) < 4:
                        print("Error: process requires at least 3 arguments")
                        print("Usage: process <input> <source> <target> [output]")
                        continue

                    input_audio = parts[1]
                    source_onoma = parts[2]
                    target_onoma = parts[3]
                    output_audio = parts[4] if len(parts) > 4 else None

                    if not os.path.exists(input_audio):
                        print(f"Error: File not found: {input_audio}")
                        continue

                    self.process_audio(input_audio, source_onoma, target_onoma, output_audio)

                elif cmd == 'set':
                    if len(parts) < 3:
                        print("Error: set requires 2 arguments")
                        print("Usage: set <param> <value>")
                        print("Available params: model_path, scaler_path, sample_rate, lambda_att, cumulative_mode")
                        continue

                    param = parts[1]
                    value = parts[2]

                    # 数値パラメータの変換
                    if param == 'sample_rate':
                        try:
                            value = int(value)
                        except ValueError:
                            print(f"Error: {param} must be an integer")
                            continue
                    elif param == 'lambda_att':
                        try:
                            value = float(value)
                            if not 0.0 <= value <= 2.0:
                                print("Warning: lambda_att is typically between 0.0 and 1.0")
                        except ValueError:
                            print(f"Error: {param} must be a float")
                            continue
                    elif param == 'cumulative_mode':
                        if value.lower() in ['true', 'on', '1', 'yes']:
                            value = True
                        elif value.lower() in ['false', 'off', '0', 'no']:
                            value = False
                        else:
                            print("Error: cumulative_mode must be true/false or on/off")
                            continue

                    self.update_settings(**{param: value})

                elif cmd == 'settings':
                    self.show_settings()

                elif cmd == 'history':
                    limit = int(parts[1]) if len(parts) > 1 else 10
                    self.show_history(limit)

                elif cmd == 'detail':
                    if len(parts) < 2:
                        print("Error: detail requires an entry ID")
                        print("Usage: detail <id>")
                        continue

                    try:
                        entry_id = int(parts[1])
                        self.show_history_detail(entry_id)
                    except ValueError:
                        print("Error: entry ID must be an integer")

                elif cmd == 'search':
                    if len(parts) < 2:
                        print("Error: search requires a query")
                        print("Usage: search <query>")
                        continue

                    query = ' '.join(parts[1:])
                    self.search_history(query)

                elif cmd == 'replay':
                    if len(parts) < 2:
                        print("Error: replay requires an entry ID")
                        print("Usage: replay <id> [output]")
                        continue

                    try:
                        entry_id = int(parts[1])
                        new_output = parts[2] if len(parts) > 2 else None
                        self.replay_history(entry_id, new_output)
                    except ValueError:
                        print("Error: entry ID must be an integer")

                elif cmd == 'clear':
                    confirm = input("Are you sure you want to clear all history? (y/N): ")
                    if confirm.lower() == 'y':
                        self.clear_history()

                # 累積モード関連コマンド
                elif cmd == 'state':
                    self.show_cumulative_state()

                elif cmd == 'newsession':
                    if len(parts) < 2:
                        print("Error: newsession requires an input file")
                        print("Usage: newsession <input_audio>")
                        continue

                    input_audio = parts[1]
                    self.new_session(input_audio)

                elif cmd == 'undo':
                    self.undo_last_edit()

                elif cmd == 'resetcumul':
                    confirm = input("Are you sure you want to reset cumulative parameters? (y/N): ")
                    if confirm.lower() == 'y':
                        self.reset_cumulative()

                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='Pair Model CLI - Transform audio using differential onomatopoeia model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python src/pair_cli.py

  # Direct processing
  python src/pair_cli.py -i input.wav -s jiajia -t tatta

  # With custom output
  python src/pair_cli.py -i input.wav -s jiajia -t tatta -o output.wav

  # Show history
  python src/pair_cli.py --history 20

  # Show history detail
  python src/pair_cli.py --detail 5

  # Search history
  python src/pair_cli.py --search "tatta"

  # Replay history entry
  python src/pair_cli.py --replay 3
        """
    )

    parser.add_argument('-i', '--input', help='Input audio file')
    parser.add_argument('-s', '--source', help='Source onomatopoeia')
    parser.add_argument('-t', '--target', help='Target onomatopoeia')
    parser.add_argument('-o', '--output', help='Output audio file (optional)')
    parser.add_argument('--model', default='models/pair_model.pth',
                        help='Model path (default: models/pair_model.pth)')
    parser.add_argument('--scaler', default='models/pair_scaler.pkl',
                        help='Scaler path (default: models/pair_scaler.pkl)')
    parser.add_argument('-a', '--attention', type=float, default=0.5,
                        help='Attention strength lambda_att (default: 0.5, 0=off)')
    parser.add_argument('--history', type=int, metavar='N',
                        help='Show last N history entries')
    parser.add_argument('--detail', type=int, metavar='ID',
                        help='Show detail of history entry')
    parser.add_argument('--search', metavar='QUERY',
                        help='Search history')
    parser.add_argument('--replay', type=int, metavar='ID',
                        help='Replay history entry')
    parser.add_argument('--clear-history', action='store_true',
                        help='Clear all history')

    args = parser.parse_args()

    cli = PairModelCLI()

    # 設定を更新
    cli.update_settings(
        model_path=args.model,
        scaler_path=args.scaler,
        lambda_att=args.attention
    )

    # 履歴クリア
    if args.clear_history:
        confirm = input("Are you sure you want to clear all history? (y/N): ")
        if confirm.lower() == 'y':
            cli.clear_history()
        return

    # 履歴表示
    if args.history is not None:
        cli.show_history(args.history)
        return

    # 履歴詳細
    if args.detail is not None:
        cli.show_history_detail(args.detail)
        return

    # 履歴検索
    if args.search:
        cli.search_history(args.search)
        return

    # 履歴再実行
    if args.replay is not None:
        cli.replay_history(args.replay, args.output)
        return

    # 直接処理
    if args.input and args.source and args.target:
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}")
            return

        cli.process_audio(args.input, args.source, args.target, args.output)
        return

    # 引数が不完全な場合は対話モードへ
    if args.input or args.source or args.target:
        print("Error: -i, -s, -t must all be specified together")
        print("Starting interactive mode instead...\n")

    # 対話モード
    cli.interactive_mode()


if __name__ == '__main__':
    main()
