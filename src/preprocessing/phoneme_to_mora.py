"""
音素列をモーラ列に変換するモジュール
"""
from typing import List, Tuple


class PhonemeToMora:
    """音素列をモーラ列に変換するクラス"""

    def __init__(self):
        # 母音のセット
        self.vowels = {'a', 'i', 'u', 'e', 'o'}

        # 子音のセット
        self.consonants = {
            'k', 'g', 's', 'sh', 'z', 'j', 't', 'ch', 'ts', 'd',
            'n', 'ny', 'h', 'hy', 'f', 'b', 'p', 'm', 'my', 'y',
            'r', 'w', 'v',
            'ky', 'gy', 'by', 'py', 'ry'
        }

        # 特殊記号
        self.special_symbols = {'N', 'Q', 'H'}

    def convert(self, phonemes: List[str]) -> List[Tuple[str, ...]]:
        """
        音素列をモーラ列に変換する

        Args:
            phonemes: 音素のリスト（例: ['k', 'a', 'N', 'k', 'a', 'N']）

        Returns:
            モーラのリスト（例: [('k', 'a'), ('N',), ('k', 'a'), ('N',)]）
        """
        moras = []
        i = 0

        while i < len(phonemes):
            current = phonemes[i]

            # 特殊記号（N, Q, H）は単独でモーラを形成
            if current in self.special_symbols:
                moras.append((current,))
                i += 1
                continue

            # 母音単独
            if current in self.vowels:
                moras.append((current,))
                i += 1
                continue

            # 子音の場合、次の音素を確認
            if current in self.consonants:
                if i + 1 < len(phonemes) and phonemes[i + 1] in self.vowels:
                    # 子音 + 母音のモーラ
                    moras.append((current, phonemes[i + 1]))
                    i += 2
                else:
                    # 子音単独（通常はありえないが、安全のため）
                    moras.append((current,))
                    i += 1
                continue

            # 未知のパターン
            print(f"Warning: Unknown phoneme pattern at position {i}: {current}")
            moras.append((current,))
            i += 1

        return moras

    def get_mora_string(self, mora: Tuple[str, ...]) -> str:
        """
        モーラをタプルから文字列に変換

        Args:
            mora: モーラのタプル（例: ('k', 'a') or ('N',)）

        Returns:
            モーラ文字列（例: 'ka' or 'N'）
        """
        return ''.join(mora)


def test_converter():
    """変換のテスト"""
    from katakana_to_phoneme import KatakanaToPhoneme

    katakana_converter = KatakanaToPhoneme()
    mora_converter = PhonemeToMora()

    test_cases = [
        'ガンガン',
        'ゴロゴロ',
        'ズシャーーッ',
        'サラサラ',
    ]

    for katakana in test_cases:
        phonemes = katakana_converter.convert(katakana)
        moras = mora_converter.convert(phonemes)
        mora_strings = [mora_converter.get_mora_string(m) for m in moras]

        print(f"{katakana}")
        print(f"  音素列: {phonemes}")
        print(f"  モーラ列: {moras}")
        print(f"  モーラ文字列: {mora_strings}")
        print()


if __name__ == '__main__':
    test_converter()
