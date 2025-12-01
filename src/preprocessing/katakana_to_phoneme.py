"""
カタカナオノマトペを音素列に変換するモジュール
Julius互換のルールを使用
"""
import re
from typing import List


class KatakanaToPhoneme:
    """カタカナを音素列に変換するクラス"""

    def __init__(self):
        # カタカナ→音素列の変換テーブル（Julius互換）
        self.katakana_map = {
            # 清音
            'ア': ['a'], 'イ': ['i'], 'ウ': ['u'], 'エ': ['e'], 'オ': ['o'],
            'カ': ['k', 'a'], 'キ': ['k', 'i'], 'ク': ['k', 'u'], 'ケ': ['k', 'e'], 'コ': ['k', 'o'],
            'サ': ['s', 'a'], 'シ': ['sh', 'i'], 'ス': ['s', 'u'], 'セ': ['s', 'e'], 'ソ': ['s', 'o'],
            'タ': ['t', 'a'], 'チ': ['ch', 'i'], 'ツ': ['ts', 'u'], 'テ': ['t', 'e'], 'ト': ['t', 'o'],
            'ナ': ['n', 'a'], 'ニ': ['n', 'i'], 'ヌ': ['n', 'u'], 'ネ': ['n', 'e'], 'ノ': ['n', 'o'],
            'ハ': ['h', 'a'], 'ヒ': ['h', 'i'], 'フ': ['f', 'u'], 'ヘ': ['h', 'e'], 'ホ': ['h', 'o'],
            'マ': ['m', 'a'], 'ミ': ['m', 'i'], 'ム': ['m', 'u'], 'メ': ['m', 'e'], 'モ': ['m', 'o'],
            'ヤ': ['y', 'a'], 'ユ': ['y', 'u'], 'ヨ': ['y', 'o'],
            'ラ': ['r', 'a'], 'リ': ['r', 'i'], 'ル': ['r', 'u'], 'レ': ['r', 'e'], 'ロ': ['r', 'o'],
            'ワ': ['w', 'a'], 'ヲ': ['w', 'o'],
            'ン': ['N'],

            # 濁音
            'ガ': ['g', 'a'], 'ギ': ['g', 'i'], 'グ': ['g', 'u'], 'ゲ': ['g', 'e'], 'ゴ': ['g', 'o'],
            'ザ': ['z', 'a'], 'ジ': ['j', 'i'], 'ズ': ['z', 'u'], 'ゼ': ['z', 'e'], 'ゾ': ['z', 'o'],
            'ダ': ['d', 'a'], 'ヂ': ['j', 'i'], 'ヅ': ['z', 'u'], 'デ': ['d', 'e'], 'ド': ['d', 'o'],
            'バ': ['b', 'a'], 'ビ': ['b', 'i'], 'ブ': ['b', 'u'], 'ベ': ['b', 'e'], 'ボ': ['b', 'o'],

            # 半濁音
            'パ': ['p', 'a'], 'ピ': ['p', 'i'], 'プ': ['p', 'u'], 'ペ': ['p', 'e'], 'ポ': ['p', 'o'],

            # 拗音（清音）
            'キャ': ['ky', 'a'], 'キュ': ['ky', 'u'], 'キョ': ['ky', 'o'],
            'シャ': ['sh', 'a'], 'シュ': ['sh', 'u'], 'シェ': ['sh', 'e'], 'ショ': ['sh', 'o'],
            'チャ': ['ch', 'a'], 'チュ': ['ch', 'u'], 'チェ': ['ch', 'e'], 'チョ': ['ch', 'o'],
            'ニャ': ['ny', 'a'], 'ニュ': ['ny', 'u'], 'ニョ': ['ny', 'o'],
            'ヒャ': ['hy', 'a'], 'ヒュ': ['hy', 'u'], 'ヒョ': ['hy', 'o'],
            'ミャ': ['my', 'a'], 'ミュ': ['my', 'u'], 'ミョ': ['my', 'o'],
            'リャ': ['ry', 'a'], 'リュ': ['ry', 'u'], 'リョ': ['ry', 'o'],

            # 拗音（濁音）
            'ギャ': ['gy', 'a'], 'ギュ': ['gy', 'u'], 'ギョ': ['gy', 'o'],
            'ジャ': ['j', 'a'], 'ジュ': ['j', 'u'], 'ジェ': ['j', 'e'], 'ジョ': ['j', 'o'],
            'ビャ': ['by', 'a'], 'ビュ': ['by', 'u'], 'ビョ': ['by', 'o'],

            # 拗音（半濁音）
            'ピャ': ['py', 'a'], 'ピュ': ['py', 'u'], 'ピョ': ['py', 'o'],

            # その他拗音
            'ファ': ['f', 'a'], 'フィ': ['f', 'i'], 'フェ': ['f', 'e'], 'フォ': ['f', 'o'],
            'ウィ': ['w', 'i'], 'ウェ': ['w', 'e'], 'ウォ': ['w', 'o'],
            'ヴァ': ['v', 'a'], 'ヴィ': ['v', 'i'], 'ヴ': ['v', 'u'], 'ヴェ': ['v', 'e'], 'ヴォ': ['v', 'o'],
            'ティ': ['t', 'i'], 'ディ': ['d', 'i'],
            'トゥ': ['t', 'u'], 'ドゥ': ['d', 'u'],

            # 促音と長音
            'ッ': ['Q'],
            'ー': ['H'],
        }

    def normalize_katakana(self, text: str) -> str:
        """
        カタカナを正規化する
        - ひらがなをカタカナに変換
        - 全角記号を正規化
        """
        # ひらがなをカタカナに変換
        normalized = ''
        for char in text:
            code = ord(char)
            # ひらがな（ぁ-ん）をカタカナに変換
            if 0x3041 <= code <= 0x3096:
                normalized += chr(code + 0x60)
            else:
                normalized += char

        return normalized

    def convert(self, katakana: str) -> List[str]:
        """
        カタカナを音素列に変換する

        Args:
            katakana: カタカナ文字列

        Returns:
            音素のリスト（例: ['k', 'a', 'N', 'k', 'a', 'N']）
        """
        # 正規化
        katakana = self.normalize_katakana(katakana)

        phonemes = []
        i = 0

        while i < len(katakana):
            # 3文字パターンを試す（拗音など）
            if i + 2 < len(katakana):
                three_char = katakana[i:i+3]
                if three_char in self.katakana_map:
                    phonemes.extend(self.katakana_map[three_char])
                    i += 3
                    continue

            # 2文字パターンを試す
            if i + 1 < len(katakana):
                two_char = katakana[i:i+2]
                if two_char in self.katakana_map:
                    phonemes.extend(self.katakana_map[two_char])
                    i += 2
                    continue

            # 1文字パターン
            one_char = katakana[i]
            if one_char in self.katakana_map:
                phonemes.extend(self.katakana_map[one_char])
            else:
                # 未知の文字は無視（または警告）
                print(f"Warning: Unknown character '{one_char}' at position {i}")

            i += 1

        return phonemes


def test_converter():
    """変換のテスト"""
    converter = KatakanaToPhoneme()

    test_cases = [
        'ガンガン',
        'ゴロゴロ',
        'ズシャーーッ',
        'サラサラ',
        'ピュウ',
    ]

    for katakana in test_cases:
        phonemes = converter.convert(katakana)
        print(f"{katakana} → {phonemes}")


if __name__ == '__main__':
    test_converter()
