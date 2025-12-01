"""
オノマトペから38次元の特徴量を抽出するモジュール
"""
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter


class OnomatopoeiaFeatureExtractor:
    """オノマトペ特徴量抽出クラス"""

    def __init__(self):
        # 母音のセット
        self.vowels = {'a', 'i', 'u', 'e', 'o'}

        # 子音カテゴリの定義
        self.cons_categories = {
            'voiceless_plosive': {'p', 'py', 't', 'k', 'ky', 'ty'},
            'voiced_plosive': {'b', 'by', 'd', 'g', 'gy'},
            'voiceless_fric': {'s', 'sh', 'f', 'h', 'hy'},
            'voiced_fric': {'z', 'j'},
            'nasal': {'m', 'my', 'n', 'ny', 'N'},
            'approximant': {'r', 'ry', 'w', 'y', 'v'}
        }

        # 子音カテゴリの逆引きマップ
        self.cons_to_category = {}
        for category, consonants in self.cons_categories.items():
            for cons in consonants:
                self.cons_to_category[cons] = category

    def extract_features(self, phonemes: List[str], moras: List[Tuple[str, ...]]) -> np.ndarray:
        """
        音素列とモーラ列から38次元の特徴量を抽出する

        Args:
            phonemes: 音素のリスト
            moras: モーラのリスト（タプルのリスト）

        Returns:
            38次元の特徴量ベクトル
        """
        features = []

        # グループA：全体構造・繰り返し (6次元)
        features.extend(self._extract_structure_features(phonemes, moras))

        # グループB：長さ・アクセント (4次元)
        features.extend(self._extract_length_features(phonemes, moras))

        # グループC：母音ヒストグラム (5次元)
        features.extend(self._extract_vowel_histogram(phonemes))

        # グループD：子音カテゴリ・ヒストグラム (6次元)
        features.extend(self._extract_consonant_category_histogram(phonemes))

        # グループE：子音比率のサマリ (3次元)
        features.extend(self._extract_consonant_ratio_summary(phonemes))

        # グループF：位置情報（語頭・語末） (14次元)
        features.extend(self._extract_position_features(moras))

        return np.array(features, dtype=np.float32)

    def _extract_structure_features(self, phonemes: List[str], moras: List[Tuple[str, ...]]) -> List[float]:
        """
        グループA：全体構造・繰り返し (6次元)
        - M: モーラ数
        - C_count: 子音トークン数
        - V_count: 母音トークン数
        - word_repeat_count: 単語レベルの繰り返し回数
        - mora_repeat_chunk_count: 同一モーラ連続の「塊数」
        - mora_repeat_ratio: 繰り返しモーラ数 / M
        """
        M = len(moras)

        # 子音と母音の数をカウント
        C_count = sum(1 for p in phonemes if p not in self.vowels and p not in {'N', 'Q', 'H'})
        V_count = sum(1 for p in phonemes if p in self.vowels)

        # モーラ文字列のリスト
        mora_strings = [''.join(m) for m in moras]

        # word_repeat_count: 単語レベルの繰り返し
        word_repeat_count = self._detect_word_repeat(mora_strings)

        # mora_repeat_chunk_count: 同一モーラ連続の塊数
        mora_repeat_chunk_count = self._count_repeat_chunks(mora_strings)

        # mora_repeat_ratio: 繰り返しモーラ数 / M
        repeat_mora_count = self._count_repeated_moras(mora_strings)
        mora_repeat_ratio = repeat_mora_count / M if M > 0 else 0.0

        return [
            float(M),
            float(C_count),
            float(V_count),
            float(word_repeat_count),
            float(mora_repeat_chunk_count),
            mora_repeat_ratio
        ]

    def _detect_word_repeat(self, mora_strings: List[str]) -> int:
        """単語レベルの繰り返し回数を検出"""
        n = len(mora_strings)
        if n == 0:
            return 1

        # 半分の長さから1まで試す
        for length in range(n // 2, 0, -1):
            pattern = mora_strings[:length]
            count = 1
            i = length

            while i + length <= n:
                if mora_strings[i:i+length] == pattern:
                    count += 1
                    i += length
                else:
                    break

            if i >= n and count > 1:
                return count

        return 1

    def _count_repeat_chunks(self, mora_strings: List[str]) -> int:
        """同一モーラ連続の塊数をカウント"""
        if not mora_strings:
            return 0

        chunk_count = 0
        i = 0

        while i < len(mora_strings):
            if i + 1 < len(mora_strings) and mora_strings[i] == mora_strings[i + 1]:
                chunk_count += 1
                # 同じモーラが続く限りスキップ
                current_mora = mora_strings[i]
                while i < len(mora_strings) and mora_strings[i] == current_mora:
                    i += 1
            else:
                i += 1

        return chunk_count

    def _count_repeated_moras(self, mora_strings: List[str]) -> int:
        """繰り返されたモーラの数をカウント"""
        if not mora_strings:
            return 0

        repeated_count = 0
        i = 0

        while i < len(mora_strings):
            if i + 1 < len(mora_strings) and mora_strings[i] == mora_strings[i + 1]:
                current_mora = mora_strings[i]
                count = 0
                while i < len(mora_strings) and mora_strings[i] == current_mora:
                    count += 1
                    i += 1
                repeated_count += count
            else:
                i += 1

        return repeated_count

    def _extract_length_features(self, phonemes: List[str], moras: List[Tuple[str, ...]]) -> List[float]:
        """
        グループB：長さ・アクセント (4次元)
        - Q_count: 促音（Q）の数
        - H_mora_count: 長音モーラ数（H または同母音連続）
        - H_ratio: 長音モーラ数 / M
        - ending_is_long: 語末が長音かどうか（0/1）
        """
        M = len(moras)
        Q_count = sum(1 for p in phonemes if p == 'Q')
        H_mora_count = sum(1 for m in moras if 'H' in m)

        # 同母音連続もカウント
        for i in range(len(moras) - 1):
            if len(moras[i]) > 0 and len(moras[i+1]) > 0:
                # 前のモーラの最後と次のモーラの最初が同じ母音
                if moras[i][-1] in self.vowels and moras[i+1][0] in self.vowels:
                    if moras[i][-1] == moras[i+1][0]:
                        H_mora_count += 1

        H_ratio = H_mora_count / M if M > 0 else 0.0

        # 語末が長音かどうか
        ending_is_long = 0.0
        if moras:
            last_mora = moras[-1]
            if 'H' in last_mora:
                ending_is_long = 1.0

        return [
            float(Q_count),
            float(H_mora_count),
            H_ratio,
            ending_is_long
        ]

    def _extract_vowel_histogram(self, phonemes: List[str]) -> List[float]:
        """
        グループC：母音ヒストグラム (5次元)
        - v_a_count, v_i_count, v_u_count, v_e_count, v_o_count
        """
        vowel_counts = Counter(p for p in phonemes if p in self.vowels)

        return [
            float(vowel_counts.get('a', 0)),
            float(vowel_counts.get('i', 0)),
            float(vowel_counts.get('u', 0)),
            float(vowel_counts.get('e', 0)),
            float(vowel_counts.get('o', 0))
        ]

    def _extract_consonant_category_histogram(self, phonemes: List[str]) -> List[float]:
        """
        グループD：子音カテゴリ・ヒストグラム (6次元)
        """
        category_counts = Counter()

        for p in phonemes:
            if p in self.cons_to_category:
                category = self.cons_to_category[p]
                category_counts[category] += 1

        return [
            float(category_counts.get('voiceless_plosive', 0)),
            float(category_counts.get('voiced_plosive', 0)),
            float(category_counts.get('voiceless_fric', 0)),
            float(category_counts.get('voiced_fric', 0)),
            float(category_counts.get('nasal', 0)),
            float(category_counts.get('approximant', 0))
        ]

    def _extract_consonant_ratio_summary(self, phonemes: List[str]) -> List[float]:
        """
        グループE：子音比率のサマリ (3次元)
        - obstruent_ratio: (破裂音 + 摩擦音のトークン数) / C_count
        - voiced_cons_ratio: (有声破裂音 + 有声音摩擦音) / C_count
        - nasal_ratio: 鼻音トークン数 / C_count
        """
        category_counts = Counter()

        for p in phonemes:
            if p in self.cons_to_category:
                category = self.cons_to_category[p]
                category_counts[category] += 1

        C_count = sum(category_counts.values())

        if C_count > 0:
            obstruent_count = (category_counts.get('voiceless_plosive', 0) +
                             category_counts.get('voiced_plosive', 0) +
                             category_counts.get('voiceless_fric', 0) +
                             category_counts.get('voiced_fric', 0))
            obstruent_ratio = obstruent_count / C_count

            voiced_cons_count = (category_counts.get('voiced_plosive', 0) +
                               category_counts.get('voiced_fric', 0))
            voiced_cons_ratio = voiced_cons_count / C_count

            nasal_count = category_counts.get('nasal', 0)
            nasal_ratio = nasal_count / C_count
        else:
            obstruent_ratio = 0.0
            voiced_cons_ratio = 0.0
            nasal_ratio = 0.0

        return [obstruent_ratio, voiced_cons_ratio, nasal_ratio]

    def _extract_position_features(self, moras: List[Tuple[str, ...]]) -> List[float]:
        """
        グループF：位置情報（語頭・語末） (14次元)
        - 語頭子音カテゴリ（6次元ワンホット）
        - 語末子音カテゴリ（6次元ワンホット）
        - starts_with_vowel（0/1）
        - ends_with_vowel（0/1）
        """
        features = []

        # 語頭のモーラ
        if moras:
            first_mora = moras[0]
            first_cons = None

            # 語頭の子音を探す
            for phoneme in first_mora:
                if phoneme in self.cons_to_category:
                    first_cons = phoneme
                    break

            # 語頭子音カテゴリのワンホット
            first_category = self.cons_to_category.get(first_cons) if first_cons else None
            for cat in ['voiceless_plosive', 'voiced_plosive', 'voiceless_fric',
                       'voiced_fric', 'nasal', 'approximant']:
                features.append(1.0 if first_category == cat else 0.0)

            # starts_with_vowel
            starts_with_vowel = 1.0 if (first_mora and first_mora[0] in self.vowels) else 0.0
        else:
            # モーラがない場合は全て0
            features.extend([0.0] * 7)
            starts_with_vowel = 0.0

        # 語末のモーラ
        if moras:
            last_mora = moras[-1]
            last_cons = None

            # 語末の子音を探す（逆順）
            for phoneme in reversed(last_mora):
                if phoneme in self.cons_to_category:
                    last_cons = phoneme
                    break

            # 語末子音カテゴリのワンホット
            last_category = self.cons_to_category.get(last_cons) if last_cons else None
            for cat in ['voiceless_plosive', 'voiced_plosive', 'voiceless_fric',
                       'voiced_fric', 'nasal', 'approximant']:
                features.append(1.0 if last_category == cat else 0.0)

            # ends_with_vowel
            ends_with_vowel = 1.0 if (last_mora and last_mora[-1] in self.vowels) else 0.0
        else:
            # モーラがない場合は全て0
            features.extend([0.0] * 7)
            ends_with_vowel = 0.0

        features.append(starts_with_vowel)
        features.append(ends_with_vowel)

        return features


def test_feature_extractor():
    """特徴量抽出のテスト"""
    from katakana_to_phoneme import KatakanaToPhoneme
    from phoneme_to_mora import PhonemeToMora

    katakana_converter = KatakanaToPhoneme()
    mora_converter = PhonemeToMora()
    feature_extractor = OnomatopoeiaFeatureExtractor()

    test_cases = [
        'ガンガン',
        'ゴロゴロ',
        'ズシャーーッ',
        'サラサラ',
    ]

    for katakana in test_cases:
        phonemes = katakana_converter.convert(katakana)
        moras = mora_converter.convert(phonemes)
        features = feature_extractor.extract_features(phonemes, moras)

        print(f"{katakana}")
        print(f"  音素列: {phonemes}")
        print(f"  モーラ列: {[''.join(m) for m in moras]}")
        print(f"  特徴量: {features.shape} - {features}")
        print()


if __name__ == '__main__':
    test_feature_extractor()
