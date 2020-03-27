#!/usr/bin/env python3

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import nltk
import os

from pypinyin import pinyin, Style
from pypinyin.style._utils import get_initials, get_finals
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin

class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass

my_pinyin = Pinyin(MyConverter())
pinyin = my_pinyin.pinyin

try:
    # For phoneme conversion, use https://github.com/Kyubyong/g2p.
    from g2p_en import G2p
    f_g2p = G2p()
    f_g2p("")
except ImportError:
    raise ImportError("g2p_en is not installed. please run `. ./path.sh && pip install g2p_en`.")
except LookupError:
    # NOTE: we need to download dict in initial running
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("punkt")


def g2p(text):
    """Convert grapheme to phoneme."""
    tokens = filter(lambda s: s != " ", f_g2p(text))
    return ' '.join(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('transcription_path', type=str, help='path for the transcription text file')
    parser.add_argument("trans_type", type=str, default="phn",
                        choices=["char", "phn"],
                        help="Input transcription type")
    args = parser.parse_args()

    # clean every line in transcription file first
    with codecs.open(args.transcription_path, 'r', 'utf-8') as fid:
        for line in fid.readlines():
            segments = line.split(" ")
            lang_char = args.transcription_path.split('/')[-1][0]
            id = segments[0] # ex. TMF1_M10001
            content = "".join(segments[1:]).replace("\n", "")

            # Some special rules to match CSMSC pinyin
            text = pinyin(content, style=Style.TONE3)
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

                    # Special rule: "e5n" -> "en5"
                    if "5" in c:
                        c = c.replace("5", "") + "5"
                    clean_content.append(c)
            print("%s %s" % (id, ' '.join(clean_content))) 

