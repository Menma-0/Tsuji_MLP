"""
Onoma2DSP CLI Tool
対話型コマンドラインインターフェース
"""
import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

# プロジェクトルートを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.onoma2dsp import Onoma2DSP


class HistoryManager:
    """編集履歴を管理するクラス"""

    def __init__(self, history_file='history/edit_history.json'):
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
                  amplification_factor, lambda_att, feature_diff_magnitude,
                  mapped_params):
        """履歴エントリを追加"""
        history = self._load_history()

        entry = {
            'id': len(history) + 1,
            'timestamp': datetime.now().isoformat(),
            'input_audio': input_audio,
            'source_onomatopoeia': source_onoma,
            'target_onomatopoeia': target_onoma,
            'output_audio': output_audio,
            'amplification_factor': amplification_factor,
            'lambda_att': lambda_att,
            'feature_diff_magnitude': feature_diff_magnitude,
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


class Onoma2DSPCLI:
    """Onoma2DSPのCLIインターフェース"""

    def __init__(self):
        self.processor = None
        self.history_manager = HistoryManager()
        self.current_settings = {
            'model_path': 'models/rwcp_model.pth',
            'scaler_path': 'models/rwcp_scaler.pkl',
            'amplification_factor': 1.0,
            'lambda_att': 0.7
        }

    def _initialize_processor(self):
        """プロセッサを初期化（遅延初期化）"""
        if self.processor is None:
            print("Loading model...")
            self.processor = Onoma2DSP(
                model_path=self.current_settings['model_path'],
                scaler_path=self.current_settings['scaler_path'],
                amplification_factor=self.current_settings['amplification_factor'],
                lambda_att=self.current_settings['lambda_att']
            )
            print("Model loaded.\n")

    def process_audio(self, input_audio, source_onoma, target_onoma, output_audio=None):
        """音声を処理"""
        self._initialize_processor()

        # 出力パスが指定されていなければ自動生成
        if output_audio is None:
            input_path = Path(input_audio)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_audio = f"output/{input_path.stem}_{source_onoma}_to_{target_onoma}_{timestamp}.wav"

        # 出力ディレクトリを作成
        os.makedirs(os.path.dirname(output_audio), exist_ok=True)

        print(f"\nProcessing:")
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
            amplification_factor=self.current_settings['amplification_factor'],
            lambda_att=self.current_settings['lambda_att'],
            feature_diff_magnitude=result['feature_diff_magnitude'],
            mapped_params=result['mapped_params']
        )

        print(f"\n[Success] Saved to: {output_audio}")
        print(f"[History] Entry #{entry_id} recorded")

        return result

    def show_history(self, limit=10):
        """履歴を表示"""
        history = self.history_manager.get_history(limit=limit)

        if not history:
            print("No history found.")
            return

        print(f"\n{'='*80}")
        print(f"Edit History (Last {len(history)} entries)")
        print(f"{'='*80}\n")

        for entry in history:
            print(f"[#{entry['id']}] {entry['timestamp']}")
            print(f"  {entry['source_onomatopoeia']} -> {entry['target_onomatopoeia']}")
            print(f"  Input:  {entry['input_audio']}")
            print(f"  Output: {entry['output_audio']}")
            print(f"  Factor: {entry['amplification_factor']}, Lambda: {entry['lambda_att']}")
            print()

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

    def show_settings(self):
        """現在の設定を表示"""
        print("\nCurrent Settings:")
        print(f"  Model:               {self.current_settings['model_path']}")
        print(f"  Scaler:              {self.current_settings['scaler_path']}")
        print(f"  Amplification Factor: {self.current_settings['amplification_factor']}")
        print(f"  Lambda (Attention):   {self.current_settings['lambda_att']}")
        print()

    def update_settings(self, **kwargs):
        """設定を更新"""
        for key, value in kwargs.items():
            if key in self.current_settings:
                self.current_settings[key] = value
                print(f"Updated {key}: {value}")

        # プロセッサをリセット（次回使用時に再初期化）
        if 'amplification_factor' in kwargs or 'lambda_att' in kwargs:
            self.processor = None

    def interactive_mode(self):
        """対話モード"""
        print("="*80)
        print(" "*20 + "Onoma2DSP CLI - Interactive Mode")
        print("="*80)
        print("\nCommands:")
        print("  process <input> <source> <target> [output] - Process audio")
        print("  set <param> <value>                         - Update settings")
        print("  settings                                    - Show current settings")
        print("  history [limit]                             - Show edit history")
        print("  search <query>                              - Search history")
        print("  help                                        - Show this help")
        print("  quit / exit                                 - Exit")
        print("\nExample:")
        print("  > process input.wav チリン ゴロゴロ")
        print("  > set amplification_factor 7.0")
        print("  > history 5")
        print("="*80)

        while True:
            try:
                command = input("\n> ").strip()

                if not command:
                    continue

                parts = command.split()
                cmd = parts[0].lower()

                if cmd in ['quit', 'exit']:
                    print("Goodbye!")
                    break

                elif cmd == 'help':
                    self.interactive_mode()  # ヘルプを再表示
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
                        continue

                    param = parts[1]
                    value = parts[2]

                    # 数値パラメータの変換
                    if param in ['amplification_factor', 'lambda_att']:
                        try:
                            value = float(value)
                        except ValueError:
                            print(f"Error: {param} must be a number")
                            continue

                    self.update_settings(**{param: value})

                elif cmd == 'settings':
                    self.show_settings()

                elif cmd == 'history':
                    limit = int(parts[1]) if len(parts) > 1 else 10
                    self.show_history(limit)

                elif cmd == 'search':
                    if len(parts) < 2:
                        print("Error: search requires a query")
                        print("Usage: search <query>")
                        continue

                    query = ' '.join(parts[1:])
                    self.search_history(query)

                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"Error: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='Onoma2DSP CLI - Transform audio using onomatopoeia',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python src/cli.py

  # Direct processing
  python src/cli.py -i input.wav -s チリン -t ゴロゴロ

  # With custom output and settings
  python src/cli.py -i input.wav -s チリン -t ゴロゴロ -o output.wav -f 7.0 -a 0.5

  # Show history
  python src/cli.py --history 20

  # Search history
  python src/cli.py --search "チリン"
        """
    )

    parser.add_argument('-i', '--input', help='Input audio file')
    parser.add_argument('-s', '--source', help='Source onomatopoeia')
    parser.add_argument('-t', '--target', help='Target onomatopoeia')
    parser.add_argument('-o', '--output', help='Output audio file (optional)')
    parser.add_argument('-f', '--factor', type=float, default=1.0,
                        help='Amplification factor (default: 1.0)')
    parser.add_argument('-a', '--attention', type=float, default=0.7,
                        help='Lambda attention (default: 0.7)')
    parser.add_argument('--history', type=int, metavar='N',
                        help='Show last N history entries')
    parser.add_argument('--search', metavar='QUERY',
                        help='Search history')

    args = parser.parse_args()

    cli = Onoma2DSPCLI()

    # 設定を更新
    cli.update_settings(
        amplification_factor=args.factor,
        lambda_att=args.attention
    )

    # 履歴表示
    if args.history is not None:
        cli.show_history(args.history)
        return

    # 履歴検索
    if args.search:
        cli.search_history(args.search)
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
