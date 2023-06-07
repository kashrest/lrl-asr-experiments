"""

This script can be used for any model on Hausa data. 
Only data preprocessing and preperation utility functions.

"""


import re
import json
import sys
import os
from typing import List
from datasets import load_metric

def compute_metrics(pred, processor):
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")
    
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

def preprocess_text(transcription: str):
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

    cleaned_transcription = _remove_special_characters(transcription)
    cleaned_transcription = _normalize_diacritics(cleaned_transcription)
    return cleaned_transcription

def preprocess_texts(transcriptions: List[str]):
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

def return_fleurs_only():
    return None
    
    
"""def return_combined_dataset():
    cache_dir_fleurs = "/data/users/kashrest/lrl-asr-experiments/data/fleurs"
    cache_dir_cv_13 = cache_dir="/data/users/kashrest/lrl-asr-experiments/data/cv_13"

    fleurs_hausa_train = load_dataset("google/fleurs", "ha_ng", split="train", cache_dir=cache_dir_fleurs)
    fleurs_hausa_val = load_dataset("google/fleurs", "ha_ng", split="validation", cache_dir=cache_dir_fleurs)
    fleurs_hausa_test = load_dataset("google/fleurs", "ha_ng", split="test", cache_dir=cache_dir_fleurs)

    cv_hausa_train, cv_hausa_val, cv_hausa_test = None, None, None
    bible_train_hausa_transcription_paths, bible_val_hausa_transcription_paths, bible_test_hausa_transcription_paths = None, None, None
    bible_train_hausa_audio_paths, bible_val_hausa_audio_paths, bible_test_hausa_audio_paths = None, None, None

    if fleurs_only is False:
        cv_hausa_train = load_dataset("mozilla-foundation/common_voice_13_0", "ha", split="train", cache_dir=cache_dir_cv_13)
        cv_hausa_val = load_dataset("mozilla-foundation/common_voice_13_0", "ha", split="validation", cache_dir=cache_dir_cv_13)
        cv_hausa_test = load_dataset("mozilla-foundation/common_voice_13_0", "ha", split="test", cache_dir=cache_dir_cv_13)

        cv_hausa_train = cv_hausa_train.cast_column("audio", Audio(sampling_rate=16_000))
        cv_hausa_val = cv_hausa_val.cast_column("audio", Audio(sampling_rate=16_000))
        cv_hausa_test = cv_hausa_test.cast_column("audio", Audio(sampling_rate=16_000))

        bible_train_hausa_transcription_paths = []
        bible_train_hausa_audio_paths = []

        bible_val_hausa_transcription_paths = []
        bible_val_hausa_audio_paths = []

        bible_test_hausa_transcription_paths = []
        bible_test_hausa_audio_paths = []

        for root, dirs, files in os.walk("./data/open_slr_129/hausa/train"):
            if len(files) > 0:
                for file in files:
                    if file[-3:] == "txt":
                        bible_train_hausa_transcription_paths.append(root+"/"+file)
                    elif file [-4:] == "flac":
                        bible_train_hausa_audio_paths.append(root+"/"+file)

        for root, dirs, files in os.walk("./data/open_slr_129/hausa/dev"):
            if len(files) > 0:
                for file in files:
                    if file[-3:] == "txt":
                        bible_val_hausa_transcription_paths.append(root+"/"+file)
                    elif file [-4:] == "flac":
                        bible_val_hausa_audio_paths.append(root+"/"+file)

        for root, dirs, files in os.walk("./data/open_slr_129/hausa/test"):
            if len(files) > 0:
                for file in files:
                    if file[-3:] == "txt":
                        bible_test_hausa_transcription_paths.append(root+"/"+file)
                    elif file [-4:] == "flac":
                        bible_test_hausa_audio_paths.append(root+"/"+file)

    model_sampling_rate = 16000

    train_audio_hausa = []
    train_transcriptions_hausa = []

    for elem in fleurs_hausa_train:
        assert elem["audio"]["sampling_rate"] == model_sampling_rate
        train_audio_hausa.append(elem["audio"]["array"])
        train_transcriptions_hausa.append(elem["raw_transcription"])

    if fleurs_only is False:
        for elem in cv_hausa_train:
            assert elem["audio"]["sampling_rate"] == model_sampling_rate
            train_audio_hausa.append(elem["audio"]["array"])
            train_transcriptions_hausa.append(elem["sentence"])

        for audio_file, transcription_file in zip(sorted(bible_train_hausa_audio_paths), sorted(bible_train_hausa_transcription_paths)):
            assert audio_file[:-5] == transcription_file[:-4]
            with open(transcription_file, "r") as f:
                transcript = f.readline()
                train_transcriptions_hausa.append(transcript)

            waveform, sample_rate = torchaudio.load(audio_file)
            resampler = T.Resample(sample_rate, model_sampling_rate, dtype=waveform.dtype)
            resampled_waveform = resampler(waveform)
            train_audio_hausa.append(resampled_waveform[0].numpy())

    val_audio_hausa = []
    val_transcriptions_hausa = []

    for elem in load_dataset("google/fleurs", "ha_ng", split="validation", cache_dir=cache_dir):
        assert elem["audio"]["sampling_rate"] == model_sampling_rate
        val_audio_hausa.append(elem["audio"]["array"])
        val_transcriptions_hausa.append(elem["raw_transcription"])

    if fleurs_only is False:
        for elem in cv_hausa_val:
            assert elem["audio"]["sampling_rate"] == model_sampling_rate
            val_audio_hausa.append(elem["audio"]["array"])
            
    return val_audio_hausa, val_transcriptions_hausa
            val_transcriptions_hausa.append(elem["sentence"])

        for audio_file, transcription_file in zip(sorted(bible_val_hausa_audio_paths), sorted(bible_val_hausa_transcription_paths)):
            assert audio_file[:-5] == transcription_file[:-4]
            with open(transcription_file, "r") as f:
                transcript = f.readline()
                val_transcriptions_hausa.append(transcript)

            waveform, sample_rate = torchaudio.load(audio_file)
            resampler = T.Resample(sample_rate, model_sampling_rate, dtype=waveform.dtype)
            resampled_waveform = resampler(waveform)
            val_audio_hausa.append(resampled_waveform[0].numpy())

    test_audio_hausa = []
    test_transcriptions_hausa = []

    for elem in load_dataset("google/fleurs", "ha_ng", split="test", cache_dir=cache_dir):
        assert elem["audio"]["sampling_rate"] == model_sampling_rate
        test_audio_hausa.append(elem["audio"]["array"])
        test_transcriptions_hausa.append(elem["raw_transcription"])

    if fleurs_only is False:
        for elem in cv_hausa_test:
            assert elem["audio"]["sampling_rate"] == model_sampling_rate
            test_audio_hausa.append(elem["audio"]["array"])
            test_transcriptions_hausa.append(elem["sentence"])

        for audio_file, transcription_file in zip(sorted(bible_test_hausa_audio_paths), sorted(bible_test_hausa_transcription_paths)):
            assert audio_file[:-5] == transcription_file[:-4]
            with open(transcription_file, "r") as f:
                transcript = f.readline()
                test_transcriptions_hausa.append(transcript)

            waveform, sample_rate = torchaudio.load(audio_file)
            resampler = T.Resample(sample_rate, model_sampling_rate, dtype=waveform.dtype)
            resampled_waveform = resampler(waveform)
            test_audio_hausa.append(resampled_waveform[0].numpy())"""