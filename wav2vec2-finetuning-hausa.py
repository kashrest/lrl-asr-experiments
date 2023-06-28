#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import matplotlib.pyplot as plt
import json

from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer

import IPython.display as ipd
import numpy as np
import random
import os
import torch
import torchaudio
import torchaudio.transforms as T
import time
import re
import sys

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import load_metric, load_dataset, Audio


# # Global Variables

# In[ ]:


assert len(sys.argv) == 13

#python wav2vec2-finetuning-hausa "facebook/mms-1b" "hausa-combined-hyperparameter-1" 16 3e-4 10 0.1 0.1 0.0 0.05 0.1 500 combined
#python baseline_asr_proof-of-concept.py model_card experiment_number batch_size learning_rate num_train_epochs attention_dropout hidden_dropout feat_proj_dropout mask_time_prob layerdrop warmup_steps fluers
#python baseline_asr_proof-of-concept.py "facebook/wav2vec2-xls-r-300m" "hausa-combined-hyperparameter-1" 16 3e-4 30 0.1 0.1 0.0 0.05 0.1 500 fleurs
#python baseline_asr_proof-of-concept.py "facebook/wav2vec2-xls-r-300m" "hausa-fleurs-only-hyperparameter-1" 16 3e-4 30 0.1 0.1 0.0 0.05 0.1 500 combined


# In[ ]:


root = "/data/users/kashrest/lrl-asr-experiments/"

pretrained_model_card = sys.argv[1]

training_experiment_number = sys.argv[2]

fleurs_only = True if sys.argv[12] == "fleurs" else False

out_dir = root+pretrained_model_card.replace("/", "_")+"/"

try:
    os.mkdir(out_dir)
except:
    print(f"Experiment folder already exists") 

out_dir = out_dir+training_experiment_number+"/"

try:
    os.mkdir(out_dir)
except:
    print(f"Experiment folder already exists") 
    
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
        test_audio_hausa.append(resampled_waveform[0].numpy())


# In[ ]:


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


# ## Character Vocabulary -- double check normalization from FLEURS
# 

# In[ ]:


chars_to_remove_regex = '[\,\?\!\-\;\:\"\“\%\‘\'\ʻ\”\�\$\&\(\)\–\—]'

def remove_special_characters(transcription):
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

def normalize_diacritics(transcription):
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

cleaned_train_transcriptions = map(remove_special_characters, train_transcriptions_hausa)
cleaned_train_transcriptions = list(map(normalize_diacritics, list(cleaned_train_transcriptions)))

cleaned_val_transcriptions = map(remove_special_characters, val_transcriptions_hausa)
cleaned_val_transcriptions = list(map(normalize_diacritics, list(cleaned_val_transcriptions)))

cleaned_test_transcriptions = map(remove_special_characters, test_transcriptions_hausa)
cleaned_test_transcriptions = list(map(normalize_diacritics, list(cleaned_test_transcriptions)))


# In[ ]:


# Character vocabulary code from: https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb#scrollTo=_0kRndSvqaKk
def extract_all_chars(transcription):
  all_text = " ".join(transcription)
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = list(map(extract_all_chars, cleaned_train_transcriptions))
vocab_val = list(map(extract_all_chars, cleaned_val_transcriptions))
vocab_test = list(map(extract_all_chars, cleaned_test_transcriptions))


# In[ ]:


vocab_train_chars = []
for elem in [elem["vocab"][0] for elem in vocab_train]:
    vocab_train_chars.extend(elem)

vocab_val_chars = []
for elem in [elem["vocab"][0] for elem in vocab_val]:
    vocab_val_chars.extend(elem)

vocab_test_chars = []
for elem in [elem["vocab"][0] for elem in vocab_test]:
    vocab_test_chars.extend(elem)


# In[ ]:


vocab_list = list(set(vocab_train_chars) | set(vocab_val_chars) | set(vocab_test_chars))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}

# for word delimiter, change " " --> "|" (ex. "Hello my name is Kaleen" --> "Hello|my|name|is|Kaleen")
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict) # this is for CTC to predict the end of a character (e.g. "hhh[PAD]iiiiii[PAD]" == "hi")

#print(f"Vocabulary length = {len(vocab_dict)} characters")
#vocab_dict


# In[ ]:


# Save vocabulary file
hausa_vocab_file = out_dir+"vocab_hausa_combined_train_val_test.json"
with open(hausa_vocab_file, 'w+') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer(hausa_vocab_file, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

assert model_sampling_rate == 16000

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=model_sampling_rate, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# In[ ]:


train_dataset = ASRDataset(train_audio_hausa, cleaned_train_transcriptions, model_sampling_rate, processor)
val_dataset = ASRDataset(val_audio_hausa, cleaned_val_transcriptions, model_sampling_rate, processor)
test_dataset = ASRDataset(test_audio_hausa, cleaned_test_transcriptions, model_sampling_rate, processor)


# In[ ]:


len(train_dataset)


# In[ ]:


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

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


# In[ ]:


wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}


# # Training

# In[ ]:


batch_size = int(sys.argv[3])
learning_rate = float(sys.argv[4])
num_train_epochs = int(sys.argv[5])
attention_dropout = float(sys.argv[6])
hidden_dropout = float(sys.argv[7])
feat_proj_dropout = float(sys.argv[8])
mask_time_prob = float(sys.argv[9])
layerdrop = float(sys.argv[10])
warmup_steps = int(sys.argv[11])
    
model = Wav2Vec2ForCTC.from_pretrained(
    pretrained_model_card, 
    attention_dropout=attention_dropout,
    hidden_dropout=hidden_dropout,
    feat_proj_dropout=feat_proj_dropout,
    mask_time_prob=mask_time_prob,
    layerdrop=layerdrop,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.num_parameters() 
model.freeze_feature_extractor()
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
  output_dir=out_dir,
  group_by_length=True,
  per_device_train_batch_size=batch_size,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=num_train_epochs,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  load_best_model_at_end=True,
  learning_rate=learning_rate,
  warmup_steps=warmup_steps,
  save_total_limit=2,
  metric_for_best_model="wer",
  greater_is_better=False
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset, 
    tokenizer=processor.feature_extractor,
)

start = time.time()
trainer.train()
end = time.time()

print(f"Training took {end-start} seconds")

hyperparameters_file = out_dir+"hyperparameters.jsonl"
with open(hyperparameters_file, "w") as f:
    obj = {"training batch size": batch_size,
           "learning rate": learning_rate,
           "number of training epochs": num_train_epochs,
           "attention dropout probability": attention_dropout,
           "hidden layer dropout probability": hidden_dropout,
           "feature projection layer dropout probability": feat_proj_dropout,
           "mask time probability": mask_time_prob,
           "layer dropout probability": layerdrop,
           "warm up steps": warmup_steps,
           "training time in seconds": end-start}
    json.dump(obj, f)


# In[ ]:


def _output_evaluation_metrics(processor, trainer, split_dataset: ASRDataset, split_dataset_str, file):
    evaluation_output = {"dataset-info": f"For '{split_dataset_str}' dataset, with trainset length {len(split_dataset)}"}

    preds = trainer.predict(split_dataset)
    eval_preds = compute_metrics(preds)

    evaluation_output["metrics"] = eval_preds
    with open(file, "w+") as f:
        json.dump(evaluation_output, f)
    
    
    pred_logits = preds.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    preds.label_ids[preds.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.batch_decode(preds.label_ids, group_tokens=False, skip_special_tokens=True)

    with open(out_dir+"trained_model_preds_"+split_dataset_str+".jsonl", "w+") as f:
        for pred in pred_strs:
            json.dump(pred, f)
            f.write("\n")

    with open(out_dir+"trained_model_gold_"+split_dataset_str+".jsonl", "w+") as f:
        for gold in label_strs:
            json.dump(gold, f)
            f.write("\n")


# In[ ]:


# Calculate metrics for each dataset in combined dataset
fleurs_hausa_val_transcriptions, fleurs_hausa_test_transcriptions = [], []
fleurs_hausa_val_audio, fleurs_hausa_test_audio = [], []

cv_hausa_val_transcriptions, cv_hausa_test_transcriptions = [], []
cv_hausa_val_audio, cv_hausa_test_audio = [], []

bible_hausa_val_transcriptions, bible_hausa_test_transcriptions = [], []
bible_hausa_val_audio, bible_hausa_test_audio = [], []


for elem in fleurs_hausa_val:
    assert elem["audio"]["sampling_rate"] == model_sampling_rate
    fleurs_hausa_val_audio.append(elem["audio"]["array"])
    fleurs_hausa_val_transcriptions.append(elem["raw_transcription"])

for elem in fleurs_hausa_test:
    assert elem["audio"]["sampling_rate"] == model_sampling_rate
    fleurs_hausa_test_audio.append(elem["audio"]["array"])
    fleurs_hausa_test_transcriptions.append(elem["raw_transcription"])

cleaned_val_transcriptions_fleurs = list(map(normalize_diacritics, list(map(remove_special_characters, fleurs_hausa_val_transcriptions))))
cleaned_test_transcriptions_fleurs = list(map(normalize_diacritics, list(map(remove_special_characters, fleurs_hausa_test_transcriptions))))

val_dataset_fleurs = ASRDataset(fleurs_hausa_val_audio, cleaned_val_transcriptions_fleurs, model_sampling_rate, processor)
_output_evaluation_metrics(processor, trainer, val_dataset_fleurs, "fleurs_validation", out_dir+"trained_model_predicted_fleurs_val-metrics.jsonl")

test_dataset_fleurs = ASRDataset(fleurs_hausa_test_audio, cleaned_test_transcriptions_fleurs, model_sampling_rate, processor)
_output_evaluation_metrics(processor, trainer, test_dataset_fleurs, "fleurs_test", out_dir+"trained_model_predicted_fleurs_test-metrics.jsonl")

if fleurs_only is False:   
    for elem in cv_hausa_val:
        assert elem["audio"]["sampling_rate"] == model_sampling_rate
        cv_hausa_val_audio.append(elem["audio"]["array"])
        cv_hausa_val_transcriptions.append(elem["sentence"])

    for elem in cv_hausa_test:
        assert elem["audio"]["sampling_rate"] == model_sampling_rate
        cv_hausa_test_audio.append(elem["audio"]["array"])
        cv_hausa_test_transcriptions.append(elem["sentence"])

    cleaned_val_transcriptions_cv = list(map(normalize_diacritics, list(map(remove_special_characters, cv_hausa_val_transcriptions))))
    cleaned_test_transcriptions_cv = list(map(normalize_diacritics, list(map(remove_special_characters, cv_hausa_test_transcriptions))))

    val_dataset_cv = ASRDataset(cv_hausa_val_audio, cleaned_val_transcriptions_cv, model_sampling_rate, processor)
    _output_evaluation_metrics(processor, trainer, val_dataset_cv, "cv_validation", out_dir+"trained_model_predicted_cv_val-metrics.jsonl")

    test_dataset_cv = ASRDataset(cv_hausa_test_audio, cleaned_test_transcriptions_cv, model_sampling_rate, processor)
    _output_evaluation_metrics(processor, trainer, test_dataset_cv, "cv_test", out_dir+"trained_model_predicted_cv_test-metrics.jsonl")


    for audio_file, transcription_file in zip(sorted(bible_val_hausa_audio_paths), sorted(bible_val_hausa_transcription_paths)):
        assert audio_file[:-5] == transcription_file[:-4]
        with open(transcription_file, "r") as f:
            transcript = f.readline()
            bible_hausa_val_transcriptions.append(transcript)

        waveform, sample_rate = torchaudio.load(audio_file)
        resampler = T.Resample(sample_rate, model_sampling_rate, dtype=waveform.dtype)
        resampled_waveform = resampler(waveform)
        bible_hausa_val_audio.append(resampled_waveform[0].numpy())
    
    for audio_file, transcription_file in zip(sorted(bible_test_hausa_audio_paths), sorted(bible_test_hausa_transcription_paths)):
        assert audio_file[:-5] == transcription_file[:-4]
        with open(transcription_file, "r") as f:
            transcript = f.readline()
            bible_hausa_test_transcriptions.append(transcript)

        waveform, sample_rate = torchaudio.load(audio_file)
        resampler = T.Resample(sample_rate, model_sampling_rate, dtype=waveform.dtype)
        resampled_waveform = resampler(waveform)
        bible_hausa_test_audio.append(resampled_waveform[0].numpy())
    
    
    cleaned_val_transcriptions_bible = list(map(normalize_diacritics, list(map(remove_special_characters, bible_hausa_val_transcriptions))))
    cleaned_test_transcriptions_bible = list(map(normalize_diacritics, list(map(remove_special_characters, bible_hausa_test_transcriptions))))    
        
    val_dataset_bible = ASRDataset(bible_hausa_val_audio, cleaned_val_transcriptions_bible, model_sampling_rate, processor)
    _output_evaluation_metrics(processor, trainer, val_dataset_bible, "bible-tts_validation", out_dir+"trained_model_predicted_bible-tts_val-metrics.jsonl")

    test_dataset_bible = ASRDataset(bible_hausa_test_audio, cleaned_test_transcriptions_bible, model_sampling_rate, processor)
    _output_evaluation_metrics(processor, trainer, test_dataset_bible, "bible-tts_test", out_dir+"trained_model_predicted_bible-tts_test-metrics.jsonl")


# In[ ]:


"""# Evaluation and predictions

evaluation_output = {"dataset-info": f"For '{fleurs_only}' dataset, with trainset length {len(train_dataset)}"}

preds_val = trainer.predict(val_dataset)
eval_val = compute_metrics(preds_val)

evaluation_output["validation set metrics"] = eval_val

preds_test = trainer.predict(test_dataset)
eval_test = compute_metrics(preds_test)

evaluation_output["test set metrics"] = eval_test

with open(out_dir+"trained_model_evaluation_metrics.jsonl", "w") as f:
    json.dump(evaluation_output, f)"""


# In[ ]:


"""pred_logits = preds_val.predictions
pred_ids = np.argmax(pred_logits, axis=-1)

preds_val.label_ids[preds_val.label_ids == -100] = processor.tokenizer.pad_token_id

pred_strs = processor.batch_decode(pred_ids)
# we do not want to group tokens when computing the metrics
label_strs = processor.batch_decode(preds_val.label_ids, group_tokens=False)

with open(out_dir+"trained_model_predicted_val.jsonl", "w") as f:
    for pred in pred_strs:
        json.dump(pred, f)
        f.write("\n")
        
with open(out_dir+"trained_model_gold_val.jsonl", "w") as f:
    for gold in label_strs:
        json.dump(gold, f)
        f.write("\n")"""


# In[ ]:


"""pred_logits = preds_test.predictions
pred_ids = np.argmax(pred_logits, axis=-1)

preds_test.label_ids[preds_test.label_ids == -100] = processor.tokenizer.pad_token_id

pred_strs = processor.batch_decode(pred_ids)
# we do not want to group tokens when computing the metrics
label_strs = processor.batch_decode(preds_test.label_ids, group_tokens=False)

with open(out_dir+"trained_model_predicted_test.jsonl", "w") as f:
    for pred in pred_strs:
        json.dump(pred, f)
        f.write("\n")
        
with open(out_dir+"trained_model_gold_test.jsonl", "w") as f:
    for gold in label_strs:
        json.dump(gold, f)
        f.write("\n")"""

