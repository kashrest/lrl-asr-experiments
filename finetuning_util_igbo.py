"""

This script can be used for any model on Hausa data. 
Only data preprocessing and preperation utility functions.

"""


import re
import json
import sys
import os
from typing import List, Union, Optional, Dict
from datasets import load_metric, load_dataset, Audio
import torch
import torchaudio
import torchaudio.transforms as T
from dataclasses import dataclass, field
from transformers import Wav2Vec2Processor


model_sampling_rate = 16000

def preprocess_text_igbo(transcription: str):
    """
    Preprocessing done according to the annotation guidelines from the 2019 LDC Igbo dataset (LDC2019S16 - IARPA Babel Igbo 
    Language Pack)
    """
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

    cleaned_transcription = _remove_special_characters(transcription)
    cleaned_transcription = _normalize_diacritics(cleaned_transcription)
    return cleaned_transcription

def preprocess_texts_igbo(transcriptions: List[str]):
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