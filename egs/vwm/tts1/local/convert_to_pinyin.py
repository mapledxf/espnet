#!/usr/bin/env python
# coding=utf-8
import sys
from pypinyin import pinyin, Style
from pypinyin.style._utils import get_initials, get_finals
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin

my_pinyin = Pinyin()
pinyin = my_pinyin.pinyin

x=sys.argv[1]

text = pinyin(x, style=Style.TONE3)
text = [c[0] for c in text]
clean_content = []
for c in text:
    c_init = get_initials(c, strict=True)
    c_final = get_finals(c, strict=True)
    for c in [c_init, c_final]:
        if len(c) == 0:
            continue
        c = c.replace("ü", "v")
        c = c.replace("ui", "uei")
        c = c.replace("un", "uen")
        c = c.replace("iu", "iou")
        if "5" in c:
            c = c.replace("5", "") + "5"
        clean_content.append(c)
print(' '.join(clean_content))
