import re
from typing import List


def preprocess(transcriptions: List[str]) -> str: 
    chars_to_remove_regex = '[\,\?\!\-\;\:\"\“\%\‘\'\ʻ\”\�\$\&\(\)\–\—]'

    def _remove_special_characters(transcription):
        transcription = transcription.strip()
        transcription = transcription.lower()
        transcription = re.sub(chars_to_remove_regex, '', transcription)
        transcription = re.sub("\[\]\{\}", '', transcription)
        transcription = re.sub(r'[\\]', '', transcription)
        transcription = re.sub(r'[/]', '', transcription)
        transcription = re.sub(u'[¥£°¾½²]', '', transcription)
        transcription = re.sub(u'[\+><]', '', transcription)
        #transcription = re.sub(and_sym, "and", transcription)
        return transcription

    def _normalize_diacritics(transcription):
        a = '[āăáã]'
        u = '[ūúü]'
        o = '[öõó]' 
        c = '[ç]'
        i = '[í]'
        s = '[ş]'
        e = '[é]'

        transcription = re.sub(a, "a", transcription)
        transcription = re.sub(u, "u", transcription)
        transcription = re.sub(o, "o", transcription)
        transcription = re.sub(c, "c", transcription)
        transcription = re.sub(i, "i", transcription)
        transcription = re.sub(s, "s", transcription)
        transcription = re.sub(e, "e", transcription)

        return transcription

    cleaned_transcriptions = map(_remove_special_characters, transcriptions)
    cleaned_transcriptions = list(map(_normalize_diacritics, list(cleaned_transcriptions)))
    return cleaned_transcriptions

