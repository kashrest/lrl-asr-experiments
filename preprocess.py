"""
Example transcript preprocessing script, for Igbo.

To use the finetuning.py script, you must provide the name of a Python script with a preprocess() function such as this one.
"""


import re
from typing import List

def preprocess(transcriptions: List[str]) -> List[str]:
    chars_to_remove_regex = '[><¥£°¾½²\\\+\,\?\!\-\;\:\"\“\%\‘\'\ʻ\”\�\$\&\(\)\–\—\[\]\{\}/]'

    def _remove_special_characters(transcription):
        transcription = transcription.strip()
        transcription = transcription.lower()
        transcription = re.sub(chars_to_remove_regex, '', transcription)
        return transcription

    def _normalize_diacritics(transcription):
        a = '[àáãă]'
        u = '[ùúüūụ]'
        o = '[òóõöọ]' 
        c = '[ç]'
        i = '[ìíīị]'
        s = '[şș]'
        e = '[èé]'
        l = '[ł]'
        n = '[ñǹṅ]'

        transcription = re.sub(a, "a", transcription)
        transcription = re.sub(u, "u", transcription)
        transcription = re.sub(o, "o", transcription)
        transcription = re.sub(c, "c", transcription)
        transcription = re.sub(i, "i", transcription)
        transcription = re.sub(s, "s", transcription)
        transcription = re.sub(e, "e", transcription)
        transcription = re.sub(l, "l", transcription)
        transcription = re.sub(n, "n", transcription)

        return transcription

    cleaned_transcriptions = map(_remove_special_characters, transcriptions)
    cleaned_transcriptions = list(map(_normalize_diacritics, list(cleaned_transcriptions)))
    return cleaned_transcriptions