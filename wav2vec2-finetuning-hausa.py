#!/usr/bin/env python
# coding: utf-8

# This script supports any Wav2Vec 2.0 architecture model (e.g. XLS-R, MMS) trained using the CTC algorithm if added to the Hugging Face Hub as a Wav2Vec2ForCTC object.
# 
# It also supports language-specific adapter training for MMS-1b-all.

# # Imports

# In[ ]:


import matplotlib.pyplot as plt
import json

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, TrainingArguments, Trainer

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

from finetuning_util_hausa import preprocess_texts as custom_hausa_preprocess
from finetuning_util_hausa import create_vocab_dict, create_data_collator, ASRDataset


# # Global Variables

# In[ ]:


# Example commands:

# python wav2vec2-finetuning-hausa.py model_card output_dir_name batch_size learning_rate num_epoch attention_dropout hidden_dropout feat_proj_dropout mask_time_prob layer_dropout warmup_steps fleurs_only

# python wav2vec2-finetuning-hausa.py "facebook/mms-1b" "hausa-combined-2" 16 3e-4 20 0.1 0.1 0.0 0.05 0.1 500 combined 


# In[ ]:


assert len(sys.argv) == 13


# In[ ]:


root = "/data/users/kashrest/lrl-asr-experiments/"

pretrained_model_card = sys.argv[1] 

training_experiment_number = sys.argv[2]

fleurs_only = True if sys.argv[12] == "fleurs" else False

adapters = True if (pretrained_model_card == "facebook/mms-1b-all" or pretrained_model_card == "facebook/mms-300m-all") else False

print(f"Training an adapters model = {adapters}")

out_dir = root+pretrained_model_card.replace("/", "_")+"/"+training_experiment_number+"/"

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


# ## Character Vocabulary
# 

# In[ ]:


cleaned_train_transcriptions = custom_hausa_preprocess(train_transcriptions_hausa)

cleaned_val_transcriptions = custom_hausa_preprocess(val_transcriptions_hausa)

cleaned_test_transcriptions = custom_hausa_preprocess(test_transcriptions_hausa)

vocab_dict = create_vocab_dict(cleaned_train_transcriptions, cleaned_val_transcriptions, cleaned_test_transcriptions)

# Save vocabulary file
hausa_vocab_file = out_dir+"vocab_hausa_combined_train_val_test.json"
with open(hausa_vocab_file, 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer(hausa_vocab_file, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

assert model_sampling_rate == 16000

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=model_sampling_rate, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

train_dataset = ASRDataset(train_audio_hausa, cleaned_train_transcriptions, model_sampling_rate, processor)
val_dataset = ASRDataset(val_audio_hausa, cleaned_val_transcriptions, model_sampling_rate, processor)
test_dataset = ASRDataset(test_audio_hausa, cleaned_test_transcriptions, model_sampling_rate, processor)

data_collator = create_data_collator(processor)


# In[ ]:


def compute_metrics(pred):
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
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True
)

if adapters is False:
    model.freeze_feature_extractor()
    model.gradient_checkpointing_enable()
else:
    model.init_adapter_layers()
    model.freeze_base_model()
    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

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


# In[ ]:


start = time.time()
trainer.train()
end = time.time()

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
    with open(file, "w") as f:
        json.dump(evaluation_output, f)
    
    
    pred_logits = preds.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    preds.label_ids[preds.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.batch_decode(preds.label_ids, group_tokens=False, skip_special_tokens=True)

    with open(out_dir+"trained_model_preds_"+split_dataset_str+".jsonl", "w") as f:
        for pred in pred_strs:
            json.dump(pred, f)
            f.write("\n")

    with open(out_dir+"trained_model_gold_"+split_dataset_str+".jsonl", "w") as f:
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

cleaned_val_transcriptions_fleurs = custom_hausa_preprocess(fleurs_hausa_val_transcriptions)
cleaned_test_transcriptions_fleurs = custom_hausa_preprocess(fleurs_hausa_test_transcriptions)

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
    
    
    cleaned_val_transcriptions_bible = custom_hausa_preprocess(bible_hausa_val_transcriptions)
    cleaned_test_transcriptions_bible = custom_hausa_preprocess(bible_hausa_test_transcriptions)    
        
    val_dataset_bible = ASRDataset(bible_hausa_val_audio, cleaned_val_transcriptions_bible, model_sampling_rate, processor)
    _output_evaluation_metrics(processor, trainer, val_dataset_bible, "bible-tts_validation", out_dir+"trained_model_predicted_bible-tts_val-metrics.jsonl")

    test_dataset_bible = ASRDataset(bible_hausa_test_audio, cleaned_test_transcriptions_bible, model_sampling_rate, processor)
    _output_evaluation_metrics(processor, trainer, test_dataset_bible, "bible-tts_test", out_dir+"trained_model_predicted_bible-tts_test-metrics.jsonl")


# In[ ]:




