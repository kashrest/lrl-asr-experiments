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

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
def create_data_collator(processor):
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    return data_collator

class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, audio, transcripts, sampling_rate, processor):
        self.audio = audio
        self.transcripts = transcripts
        self.sampling_rate = sampling_rate
        self.processor = processor

    def __getitem__(self, idx):
        input_values = self.processor(self.audio[idx], sampling_rate=self.sampling_rate).input_values[0]
        labels = None
        with self.processor.as_target_processor():
            labels = self.processor(self.transcripts[idx]).input_ids
        item = {}
        item["input_values"] = input_values
        item["labels"] = labels
        #item = {"audio": self.audio[idx], "transcription": self.transcripts[idx], "sampling_rate": self.sampling_rate}
        return item

    def __len__(self):
        return len(self.transcripts)

def create_vocab_dict(cleaned_train_transcriptions, cleaned_val_transcriptions, cleaned_test_transcriptions):
    # Character vocabulary code from: https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb#scrollTo=_0kRndSvqaKk
    def extract_all_chars(transcription):
      all_text = " ".join(transcription)
      vocab = list(set(all_text))
      return {"vocab": [vocab], "all_text": [all_text]}

    vocab_train = list(map(extract_all_chars, cleaned_train_transcriptions))
    vocab_val = list(map(extract_all_chars, cleaned_val_transcriptions))
    vocab_test = list(map(extract_all_chars, cleaned_test_transcriptions))
    
    vocab_train_chars = []
    for elem in [elem["vocab"][0] for elem in vocab_train]:
        vocab_train_chars.extend(elem)

    vocab_val_chars = []
    for elem in [elem["vocab"][0] for elem in vocab_val]:
        vocab_val_chars.extend(elem)

    vocab_test_chars = []
    for elem in [elem["vocab"][0] for elem in vocab_test]:
        vocab_test_chars.extend(elem)
        
    vocab_list = list(set(vocab_train_chars) | set(vocab_val_chars) | set(vocab_test_chars))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    # for word delimiter, change " " --> "|" (ex. "Hello my name is Kaleen" --> "Hello|my|name|is|Kaleen")
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict) # this is for CTC to predict the end of a character (e.g. "hhh[PAD]iiiiii[PAD]" == "hi")
    return vocab_dict

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

def return_fleurs_test():
    cache_dir_fleurs = "/data/users/kashrest/lrl-asr-experiments/data/fleurs"
    fleurs_hausa_test = load_dataset("google/fleurs", "ha_ng", split="test", cache_dir=cache_dir_fleurs)
    test_audio_hausa = []
    test_transcriptions_hausa = []

    for elem in fleurs_hausa_test:
        assert elem["audio"]["sampling_rate"] == model_sampling_rate
        test_audio_hausa.append(elem["audio"]["array"])
        test_transcriptions_hausa.append(elem["raw_transcription"])
    return test_audio_hausa, test_transcriptions_hausa

def return_cv_test():
    cache_dir_cv_13 = cache_dir="/data/users/kashrest/lrl-asr-experiments/data/cv_13"
    cv_hausa_test = load_dataset("mozilla-foundation/common_voice_13_0", "ha", split="test", cache_dir=cache_dir_cv_13)
    cv_hausa_test = cv_hausa_test.cast_column("audio", Audio(sampling_rate=16_000))
    test_audio_hausa = []
    test_transcriptions_hausa = []

    for elem in cv_hausa_test:
        assert elem["audio"]["sampling_rate"] == model_sampling_rate
        test_audio_hausa.append(elem["audio"]["array"])
        test_transcriptions_hausa.append(elem["sentence"])
    return test_audio_hausa, test_transcriptions_hausa

def return_bibletts():
    test_audio_hausa, test_transcriptions_hausa = [], []
    bible_test_hausa_audio_paths, bible_test_hausa_transcription_paths = [], []
    for root, dirs, files in os.walk("./data/open_slr_129/hausa/test"):
        if len(files) > 0:
            for file in files:
                if file[-3:] == "txt":
                    bible_test_hausa_transcription_paths.append(root+"/"+file)
                elif file [-4:] == "flac":
                    bible_test_hausa_audio_paths.append(root+"/"+file)
                    
    for audio_file, transcription_file in zip(sorted(bible_test_hausa_audio_paths), sorted(bible_test_hausa_transcription_paths)):
        assert audio_file[:-5] == transcription_file[:-4]
        with open(transcription_file, "r") as f:
            transcript = f.readline()
            test_transcriptions_hausa.append(transcript)

        waveform, sample_rate = torchaudio.load(audio_file)
        resampler = T.Resample(sample_rate, model_sampling_rate, dtype=waveform.dtype)
        resampled_waveform = resampler(waveform)
        test_audio_hausa.append(resampled_waveform[0].numpy())
        
    return test_audio_hausa, test_transcriptions_hausa
    
def return_combined_dataset():
    cache_dir_fleurs = "/data/users/kashrest/lrl-asr-experiments/data/fleurs"
    cache_dir_cv_13 = cache_dir="/data/users/kashrest/lrl-asr-experiments/data/cv_13"

    fleurs_hausa_train = load_dataset("google/fleurs", "ha_ng", split="train", cache_dir=cache_dir_fleurs)
    fleurs_hausa_val = load_dataset("google/fleurs", "ha_ng", split="validation", cache_dir=cache_dir_fleurs)
    fleurs_hausa_test = load_dataset("google/fleurs", "ha_ng", split="test", cache_dir=cache_dir_fleurs)

    cv_hausa_train, cv_hausa_val, cv_hausa_test = None, None, None
    bible_train_hausa_transcription_paths, bible_val_hausa_transcription_paths, bible_test_hausa_transcription_paths = None, None, None
    bible_train_hausa_audio_paths, bible_val_hausa_audio_paths, bible_test_hausa_audio_paths = None, None, None

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

    for elem in cv_hausa_val:
        assert elem["audio"]["sampling_rate"] == model_sampling_rate
        val_audio_hausa.append(elem["audio"]["array"])

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
        test_audio_hausa.append(resampled_waveform[0].numpy())
        
    return {"train": {"audio": train_audio_hausa, "transcriptions": train_transcriptions_hausa}, "validation": {"audio": val_audio_hausa, "transcriptions": val_transcriptions_hausa}, "test": {"audio": test_audio_hausa, "transcriptions": test_transcriptions_hausa}}
