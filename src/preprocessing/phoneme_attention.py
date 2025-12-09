"""
オノマトペの音韻特徴からDSPパラメータへのAttention（重み付け）を計算するモジュール

学術文献に基づく音響特徴量とオノマトペの対応関係:

1. フォルマント周波数 (F1, F2) とサイズ知覚
   - 高F2（前舌母音 /i/）: 小さいサイズ、角張り (F2 ≈ 2300 Hz)
   - 低F2（後舌母音 /a/, /o/）: 大きいサイズ、開放感 (F2 ≈ 1300 Hz)
   - Knoeferle et al. (2017), Ohala (1994)

2. スペクトルバランス（800-1800 Hz境界）
   - 低スペクトルバランス: 丸み → 低域EQ強調
   - 高スペクトルバランス: 角張り → 高域EQ強調
   - Fort & Schwartz (2022): r² = 60%

3. 時間的連続性
   - 連続的な音: 丸み、滑らかさ → 長いサスティン
   - 不連続な音（促音など）: 角張り、瞬間性 → 短いアタック
   - Fort & Schwartz (2022)

4. 高周波エネルギー (>10kHz)
   - 高い高周波エネルギー: 角張り、鋭さ → presence帯域
   - Chen et al. (2016)

5. 有声性
   - 有声子音 (/b/, /d/, /g/, /z/): 大きい、重い、粗い
   - 無声子音 (/p/, /t/, /k/, /s/): 小さい、軽い、細かい
   - Kawahara et al. (2018), Hamano (1998)

6. 子音タイプと音響的基盤
   - /p/初頭: 高周波バースト → 張り、軽さ、小ささ
   - /b/初頭: 低周波有声成分 → 重さ、大きさ、鈍さ
   - /k/初頭: 急峻なバースト → 硬さ、鋭さ、突然性
   - /g/初頭: 有声+バースト → 重さ、粗さ、強度
   - /s/初頭: 高周波摩擦雑音 → 滑らかさ、流動性
   - /z/初頭: 有声摩擦雑音 → 粗さ、振動感

7. 促音（っ）と長音（ー）
   - 促音: 急峻な時間構造 → 急速さ、瞬間性 → 短いアタック
   - 長音: 持続時間延長 → 継続性、強調 → 長いサスティン、タイムストレッチ
"""
import numpy as np
from typing import List, Tuple, Dict


class PhonemeAttention:
    """
    オノマトペの音韻特徴からDSPパラメータへのAttention重みを計算

    DSPパラメータ（10次元）:
    [0] gain          - 音量
    [1] compression   - 圧縮
    [2] eq_sub        - 超低域 (20-80 Hz)
    [3] eq_low        - 低域 (80-250 Hz)
    [4] eq_mid        - 中域 (250-2000 Hz)
    [5] eq_high       - 高域 (2000-6000 Hz)
    [6] eq_presence   - 超高域 (6000-20000 Hz)
    [7] attack        - アタック
    [8] sustain       - サスティン
    [9] time_stretch  - タイムストレッチ
    """

    def __init__(self):
        # 母音セット
        self.vowels = {'a', 'i', 'u', 'e', 'o'}

        # 前舌母音（高F2）: 小さい、鋭い → 高域強調
        self.front_vowels = {'i', 'e'}

        # 後舌母音（低F2）: 大きい、丸い → 低域強調
        self.back_vowels = {'a', 'o', 'u'}

        # 有声阻害音: 重い、大きい、粗い
        self.voiced_obstruents = {'b', 'by', 'd', 'g', 'gy', 'z', 'j'}

        # 無声阻害音: 軽い、小さい、細かい
        self.voiceless_obstruents = {'p', 'py', 't', 'k', 'ky', 'ty', 's', 'sh', 'f', 'h', 'hy'}

        # 無声破裂音: 急峻なバースト → アタック強調
        self.voiceless_plosives = {'p', 'py', 't', 'k', 'ky', 'ty'}

        # 有声破裂音: 有声+バースト → 低域、重さ
        self.voiced_plosives = {'b', 'by', 'd', 'g', 'gy'}

        # 摩擦音: 高周波成分 → presence強調
        self.fricatives = {'s', 'sh', 'f', 'h', 'hy', 'z', 'j'}

        # 高周波摩擦音（無声）: 滑らかさ、流動性
        self.voiceless_fricatives = {'s', 'sh', 'f', 'h', 'hy'}

        # 鼻音: 柔らかさ、響き
        self.nasals = {'m', 'my', 'n', 'ny', 'N'}

        # 流音・半母音: 滑らかさ
        self.approximants = {'r', 'ry', 'w', 'y'}

        # 特殊モーラ
        self.special_moras = {'Q', 'N', 'H'}

    def compute_attention(
        self,
        phonemes: List[str],
        moras: List[Tuple[str, ...]]
    ) -> np.ndarray:
        """
        音韻特徴からDSPパラメータへのAttention重みを計算

        Args:
            phonemes: 音素リスト
            moras: モーラリスト

        Returns:
            10次元のAttention重みベクトル（0-1の範囲）
        """
        # 基本重み（全て0.5からスタート）
        attention = np.ones(10) * 0.5

        # 各特徴を分析
        vowel_attention = self._analyze_vowels(phonemes)
        consonant_attention = self._analyze_consonants(phonemes)
        temporal_attention = self._analyze_temporal(phonemes, moras)
        initial_attention = self._analyze_initial_consonant(moras)

        # 重みを統合（加重平均）
        attention = (
            attention * 0.2 +
            vowel_attention * 0.25 +
            consonant_attention * 0.25 +
            temporal_attention * 0.15 +
            initial_attention * 0.15
        )

        # 0-1にクリップ
        attention = np.clip(attention, 0.0, 1.0)

        return attention

    def _analyze_vowels(self, phonemes: List[str]) -> np.ndarray:
        """
        母音分析: F2フォルマントとサイズ/形状の対応

        - 前舌母音 /i/, /e/: 高F2 → 小さい、角張り → 高域EQ、presence
        - 後舌母音 /a/, /o/, /u/: 低F2 → 大きい、丸み → 低域EQ
        """
        attention = np.ones(10) * 0.5

        vowel_phonemes = [p for p in phonemes if p in self.vowels]
        if not vowel_phonemes:
            return attention

        front_count = sum(1 for v in vowel_phonemes if v in self.front_vowels)
        back_count = sum(1 for v in vowel_phonemes if v in self.back_vowels)
        total = len(vowel_phonemes)

        front_ratio = front_count / total
        back_ratio = back_count / total

        # 前舌母音が多い: 高域・presence強調、低域抑制
        # [0]gain [1]comp [2]sub [3]low [4]mid [5]high [6]pres [7]atk [8]sus [9]stretch
        attention[2] = 0.5 - front_ratio * 0.3  # sub: 前舌で抑制
        attention[3] = 0.5 - front_ratio * 0.3  # low: 前舌で抑制
        attention[5] = 0.5 + front_ratio * 0.4  # high: 前舌で強調
        attention[6] = 0.5 + front_ratio * 0.4  # presence: 前舌で強調

        # 後舌母音が多い: 低域強調、高域抑制
        attention[2] += back_ratio * 0.3  # sub: 後舌で強調
        attention[3] += back_ratio * 0.4  # low: 後舌で強調
        attention[5] -= back_ratio * 0.2  # high: 後舌で抑制
        attention[6] -= back_ratio * 0.2  # presence: 後舌で抑制

        return attention

    def _analyze_consonants(self, phonemes: List[str]) -> np.ndarray:
        """
        子音分析: 有声性と子音タイプの対応

        - 有声阻害音: 重い、大きい → 低域、圧縮
        - 無声阻害音: 軽い、小さい → 高域、アタック
        - 摩擦音: 高周波 → presence
        - 鼻音: 柔らかさ → 中域
        """
        attention = np.ones(10) * 0.5

        consonants = [p for p in phonemes if p not in self.vowels and p not in self.special_moras]
        if not consonants:
            return attention

        total = len(consonants)

        voiced_obs_count = sum(1 for c in consonants if c in self.voiced_obstruents)
        voiceless_obs_count = sum(1 for c in consonants if c in self.voiceless_obstruents)
        fricative_count = sum(1 for c in consonants if c in self.fricatives)
        plosive_count = sum(1 for c in consonants if c in self.voiceless_plosives or c in self.voiced_plosives)
        nasal_count = sum(1 for c in consonants if c in self.nasals)

        voiced_ratio = voiced_obs_count / total
        voiceless_ratio = voiceless_obs_count / total
        fricative_ratio = fricative_count / total
        plosive_ratio = plosive_count / total
        nasal_ratio = nasal_count / total

        # 有声阻害音: 重い、大きい → 低域強調、圧縮
        attention[1] = 0.5 + voiced_ratio * 0.4  # compression
        attention[2] = 0.5 + voiced_ratio * 0.3  # sub
        attention[3] = 0.5 + voiced_ratio * 0.4  # low
        attention[0] = 0.5 + voiced_ratio * 0.2  # gain（重さ）

        # 無声阻害音: 軽い、小さい → 高域、アタック強調
        attention[5] += voiceless_ratio * 0.3  # high
        attention[6] += voiceless_ratio * 0.3  # presence
        attention[7] = 0.5 + voiceless_ratio * 0.3  # attack

        # 摩擦音: 高周波成分 → presence帯域
        attention[6] += fricative_ratio * 0.4  # presence
        attention[8] = 0.5 + fricative_ratio * 0.2  # sustain（連続性）

        # 破裂音: 急峻なバースト → アタック
        attention[7] += plosive_ratio * 0.4  # attack

        # 鼻音: 柔らかさ → 中域、サスティン
        attention[4] = 0.5 + nasal_ratio * 0.3  # mid
        attention[8] += nasal_ratio * 0.3  # sustain

        return attention

    def _analyze_temporal(self, phonemes: List[str], moras: List[Tuple[str, ...]]) -> np.ndarray:
        """
        時間的特徴分析: 促音・長音の対応

        - 促音（Q）: 急峻な時間構造 → 短いアタック、瞬間性
        - 長音（H/-）: 持続時間延長 → 長いサスティン、タイムストレッチ
        """
        attention = np.ones(10) * 0.5

        mora_count = len(moras)
        if mora_count == 0:
            return attention

        # 促音のカウント
        q_count = sum(1 for p in phonemes if p == 'Q')
        # 長音のカウント（H または 撥音N）
        h_count = sum(1 for p in phonemes if p == 'H')
        # 撥音
        n_count = sum(1 for p in phonemes if p == 'N')

        q_ratio = q_count / mora_count
        h_ratio = h_count / mora_count
        n_ratio = n_count / mora_count

        # 促音: 瞬間性 → アタック強調、サスティン抑制、タイムストレッチ短縮
        attention[7] = 0.5 + q_ratio * 0.5  # attack: 強調
        attention[8] = 0.5 - q_ratio * 0.4  # sustain: 抑制
        attention[9] = 0.5 - q_ratio * 0.3  # time_stretch: 短縮

        # 長音: 継続性 → サスティン強調、タイムストレッチ延長
        attention[8] += h_ratio * 0.5  # sustain: 強調
        attention[9] += h_ratio * 0.4  # time_stretch: 延長
        attention[7] -= h_ratio * 0.2  # attack: 抑制

        # 撥音: 響き → 中域、サスティン
        attention[4] += n_ratio * 0.3  # mid
        attention[8] += n_ratio * 0.3  # sustain

        return attention

    def _analyze_initial_consonant(self, moras: List[Tuple[str, ...]]) -> np.ndarray:
        """
        語頭子音分析: 初頭音の音響的基盤

        日本語オノマトペの音韻-音響-意味対応（Hamano 1998, Kawahara 2018）:
        - /p/初頭: 高周波バースト → 張り、軽さ → 高域EQ
        - /b/初頭: 低周波有声成分 → 重さ、鈍さ → 低域EQ、圧縮
        - /k/初頭: 急峻なバースト → 硬さ、鋭さ → アタック、高域
        - /g/初頭: 有声+バースト → 重さ、粗さ → 低域、圧縮
        - /s/初頭: 高周波摩擦 → 滑らかさ → presence、サスティン
        - /z/初頭: 有声摩擦 → 粗さ、振動 → 中域、presence
        - /t/初頭: 鋭いバースト → 軽さ、鋭さ → アタック
        - /d/初頭: 有声バースト → 重さ → 低域
        """
        attention = np.ones(10) * 0.5

        if not moras:
            return attention

        first_mora = moras[0]
        if not first_mora:
            return attention

        # 語頭の子音を取得
        initial_cons = None
        for phoneme in first_mora:
            if phoneme not in self.vowels and phoneme not in self.special_moras:
                initial_cons = phoneme
                break

        if initial_cons is None:
            return attention

        # 各初頭子音に対する重み設定
        # [0]gain [1]comp [2]sub [3]low [4]mid [5]high [6]pres [7]atk [8]sus [9]stretch

        if initial_cons in {'p', 'py'}:
            # /p/: 張り、軽さ、小ささ → 高域、presence
            attention[5] = 0.8  # high
            attention[6] = 0.8  # presence
            attention[7] = 0.7  # attack
            attention[3] = 0.3  # low: 抑制

        elif initial_cons in {'b', 'by'}:
            # /b/: 重さ、大きさ、鈍さ → 低域、圧縮
            attention[2] = 0.7  # sub
            attention[3] = 0.8  # low
            attention[1] = 0.7  # compression
            attention[0] = 0.7  # gain
            attention[6] = 0.3  # presence: 抑制

        elif initial_cons in {'k', 'ky'}:
            # /k/: 硬さ、鋭さ、突然性 → アタック、高域
            attention[7] = 0.9  # attack
            attention[5] = 0.7  # high
            attention[6] = 0.6  # presence
            attention[8] = 0.3  # sustain: 抑制

        elif initial_cons in {'g', 'gy'}:
            # /g/: 重さ、粗さ、強度 → 低域、圧縮
            attention[2] = 0.6  # sub
            attention[3] = 0.8  # low
            attention[1] = 0.7  # compression
            attention[7] = 0.6  # attack

        elif initial_cons in {'t', 'ty'}:
            # /t/: 軽さ、鋭さ → アタック、高域
            attention[7] = 0.8  # attack
            attention[5] = 0.7  # high
            attention[8] = 0.3  # sustain: 抑制

        elif initial_cons == 'd':
            # /d/: 重さ → 低域、アタック
            attention[3] = 0.7  # low
            attention[7] = 0.6  # attack
            attention[1] = 0.6  # compression

        elif initial_cons in {'s', 'sh'}:
            # /s/: 滑らかさ、流動性 → presence、サスティン
            attention[6] = 0.9  # presence
            attention[5] = 0.7  # high
            attention[8] = 0.7  # sustain
            attention[3] = 0.3  # low: 抑制

        elif initial_cons in {'z', 'j'}:
            # /z/: 粗さ、振動感 → 中域、presence
            attention[4] = 0.7  # mid
            attention[6] = 0.7  # presence
            attention[8] = 0.6  # sustain

        elif initial_cons in {'h', 'hy', 'f'}:
            # /h/: 息、軽さ → presence
            attention[6] = 0.8  # presence
            attention[5] = 0.6  # high
            attention[3] = 0.3  # low: 抑制

        elif initial_cons in {'m', 'my'}:
            # /m/: 柔らかさ → 中域、サスティン
            attention[4] = 0.7  # mid
            attention[8] = 0.7  # sustain

        elif initial_cons in {'n', 'ny'}:
            # /n/: 響き → 中域
            attention[4] = 0.7  # mid
            attention[8] = 0.6  # sustain

        elif initial_cons in {'r', 'ry'}:
            # /r/: 弾み → アタック、中域
            attention[7] = 0.6  # attack
            attention[4] = 0.6  # mid

        elif initial_cons == 'w':
            # /w/: 丸み → 低域
            attention[3] = 0.7  # low
            attention[8] = 0.6  # sustain

        elif initial_cons == 'y':
            # /y/: 軽さ → 高域
            attention[5] = 0.6  # high

        return attention

    def get_attention_explanation(
        self,
        phonemes: List[str],
        moras: List[Tuple[str, ...]]
    ) -> Dict[str, any]:
        """
        Attention計算の説明を生成

        Returns:
            各分析の詳細と最終的なAttention重み
        """
        attention = self.compute_attention(phonemes, moras)

        # 語頭子音の特定
        initial_cons = None
        if moras and moras[0]:
            for phoneme in moras[0]:
                if phoneme not in self.vowels and phoneme not in self.special_moras:
                    initial_cons = phoneme
                    break

        # 母音分析
        vowel_phonemes = [p for p in phonemes if p in self.vowels]
        front_count = sum(1 for v in vowel_phonemes if v in self.front_vowels)
        back_count = sum(1 for v in vowel_phonemes if v in self.back_vowels)

        # 子音分析
        consonants = [p for p in phonemes if p not in self.vowels and p not in self.special_moras]
        voiced_count = sum(1 for c in consonants if c in self.voiced_obstruents)
        voiceless_count = sum(1 for c in consonants if c in self.voiceless_obstruents)

        # 時間的特徴
        q_count = sum(1 for p in phonemes if p == 'Q')
        h_count = sum(1 for p in phonemes if p == 'H')

        param_names = [
            'gain', 'compression', 'eq_sub', 'eq_low', 'eq_mid',
            'eq_high', 'eq_presence', 'attack', 'sustain', 'time_stretch'
        ]

        return {
            'attention_weights': {name: float(attention[i]) for i, name in enumerate(param_names)},
            'analysis': {
                'initial_consonant': initial_cons,
                'vowels': {
                    'front_vowels': front_count,
                    'back_vowels': back_count,
                    'total': len(vowel_phonemes)
                },
                'consonants': {
                    'voiced_obstruents': voiced_count,
                    'voiceless_obstruents': voiceless_count,
                    'total': len(consonants)
                },
                'temporal': {
                    'geminate_count': q_count,
                    'long_vowel_count': h_count
                }
            },
            'top_attention': sorted(
                [(name, float(attention[i])) for i, name in enumerate(param_names)],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


def test_phoneme_attention():
    """PhonemeAttentionのテスト"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from katakana_to_phoneme import KatakanaToPhoneme
    from phoneme_to_mora import PhonemeToMora

    katakana_converter = KatakanaToPhoneme()
    mora_converter = PhonemeToMora()
    attention_calculator = PhonemeAttention()

    test_cases = [
        'ピカピカ',    # /p/初頭、前舌母音 → 高域、軽さ
        'ボコボコ',    # /b/初頭、後舌母音 → 低域、重さ
        'カチカチ',    # /k/初頭 → アタック、硬さ
        'ゴロゴロ',    # /g/初頭、後舌母音 → 低域、重さ
        'サラサラ',    # /s/初頭 → presence、滑らかさ
        'ザラザラ',    # /z/初頭 → 粗さ
        'ドーン',      # /d/初頭、長音 → 低域、サスティン
        'パッ',        # /p/初頭、促音 → アタック、瞬間性
        'チリンチリン', # 前舌母音多い → 高域
    ]

    print("=" * 80)
    print("PhonemeAttention Test")
    print("=" * 80)

    for katakana in test_cases:
        phonemes = katakana_converter.convert(katakana)
        moras = mora_converter.convert(phonemes)

        explanation = attention_calculator.get_attention_explanation(phonemes, moras)

        print(f"\n{katakana}")
        print(f"  Phonemes: {phonemes}")
        print(f"  Moras: {[''.join(m) for m in moras]}")
        print(f"  Initial consonant: {explanation['analysis']['initial_consonant']}")
        print(f"  Top 5 attention weights:")
        for name, weight in explanation['top_attention']:
            bar = '#' * int(weight * 20)
            print(f"    {name:<15}: {weight:.3f} {bar}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    test_phoneme_attention()
