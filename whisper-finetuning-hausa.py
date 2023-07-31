#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import load_dataset, load_metric, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, Seq2SeqTrainer

import torch
import time

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate

import torchaudio
import torchaudio.transforms as T
import re
import sys
import os
import json
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# In[ ]:


assert len(sys.argv) == 7

#python whisper-finetuning-experiments.py model_card experiment_name train_batch_size learning_rate num_train_epochs fluers
#python whisper-finetuning-experiments.py "openai/whisper-large" "fleurs_only_preprocessed_experiment_1" 16 1e-5 20 fleurs
#python whisper-finetuning-experiments.py "openai/whisper-large" "fleurs_only_preprocessed_experiment_1" 16 1e-5 20 combined


model_card = sys.argv[1]
experiment_name = sys.argv[2]
train_batch_size = int(sys.argv[3])
learning_rate = float(sys.argv[4])
num_train_epochs = int(sys.argv[5])
fleurs_only = True if sys.argv[6] == "fleurs" else False

root = "/data/users/backup/2023-07-27_kashrest/"

out_dir = root + "model_checkpoints/" + model_card.replace("/","_") + "/" + experiment_name + "/"

try:
    os.mkdir(out_dir)
except:
    print(f"Experiment folder already exists") 


# In[ ]:


cache_dir_fleurs = root + "data/fleurs"
cache_dir_cv_13 = cache_dir= root + "data/cv_13"

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

    for root, dirs, files in os.walk(root + "data/open_slr_129/hausa/train"):
        if len(files) > 0:
            for file in files:
                if file[-3:] == "txt":
                    bible_train_hausa_transcription_paths.append(root+"/"+file)
                elif file [-4:] == "flac":
                    bible_train_hausa_audio_paths.append(root+"/"+file)

    for root, dirs, files in os.walk(root + "data/open_slr_129/hausa/dev"):
        if len(files) > 0:
            for file in files:
                if file[-3:] == "txt":
                    bible_val_hausa_transcription_paths.append(root+"/"+file)
                elif file [-4:] == "flac":
                    bible_val_hausa_audio_paths.append(root+"/"+file)

    for root, dirs, files in os.walk(root + "data/open_slr_129/hausa/test"):
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

for elem in fleurs_hausa_val:
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

for elem in fleurs_hausa_test:
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


processor = WhisperProcessor.from_pretrained(model_card, language="Hausa", task="transcribe")


# In[ ]:


class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, audio, transcripts, sampling_rate, processor):#feature_extractor, tokenizer):
        self.audio = audio
        self.transcripts = transcripts
        self.sampling_rate = sampling_rate
        self.processor = processor
        """self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
"""
    def __getitem__(self, idx):
        input_values = self.processor.feature_extractor(self.audio[idx], sampling_rate=self.sampling_rate).input_features[0]
        labels = self.processor.tokenizer(self.transcripts[idx]).input_ids
        
        item = {}
        item["input_features"] = input_values
        item["labels"] = labels
        
        return item

    def __len__(self):
        return len(self.transcripts)


# In[ ]:


train_dataset = ASRDataset(train_audio_hausa, cleaned_train_transcriptions, model_sampling_rate, processor)
val_dataset = ASRDataset(val_audio_hausa, cleaned_val_transcriptions, model_sampling_rate, processor)
test_dataset = ASRDataset(test_audio_hausa, cleaned_test_transcriptions, model_sampling_rate, processor)


# In[ ]:


len(train_dataset)


# In[ ]:


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# In[ ]:


metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


# In[ ]:


model = WhisperForConditionalGeneration.from_pretrained(model_card)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir=out_dir,  # change to a repo name of your choice
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=learning_rate,
    warmup_steps=500,
    num_train_epochs=num_train_epochs,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    save_total_limit=2, 
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

start = time.time()
trainer.train()
end = time.time()

with open(out_dir+"hyperparameters.jsonl", "w+") as f:
    out = {}
    out["train_batch_size"] = train_batch_size
    out["learning_rate"] = learning_rate
    out["num_train_epochs"] = num_train_epochs
    out["training time in seconds"] = end-start
    json.dump(out, f)

print(f"Training took {end-start} seconds")


# In[ ]:


# Evaluation and predictions

"""evaluation_output = {"dataset-info": f"For '{fleurs_only}' dataset, with trainset length {len(train_dataset)}"}

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




