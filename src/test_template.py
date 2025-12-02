"""
create_dsp_templateの出力を確認
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.create_rwcp_dataset import create_dsp_template


test_onomas = [
    'チリン',
    'ゴロゴロ',
    'カッ',
    'ガッ',
    'キラキラ',
    'サラサラ',
    'ドンドン',
]

param_names = ['gain', 'comp', 'eq_sub', 'eq_low', 'eq_mid',
               'eq_high', 'eq_pres', 'atk', 'sus', 'stretch']

for onoma in test_onomas:
    print(f"\n{onoma}:")
    template = create_dsp_template(onoma)
    for i, (name, val) in enumerate(zip(param_names, template)):
        print(f"  {name:<10}: {val:>6.3f}")
