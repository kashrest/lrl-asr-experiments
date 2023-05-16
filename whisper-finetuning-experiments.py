#!/usr/bin/env python
# coding: utf-8

# In[8]:


from datasets import load_dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import json


# In[4]:

root = "/data/users/kashrest/lrl-asr-experiments/"
cache_dir=root+"data/fleurs"
pretrained_model_card = "openai/whisper-large"
training_experiment_number = "1"
out_dir = root+pretrained_model_card.replace("/", "_")+"/"+"experiment_"+training_experiment_number+"/"

try:
    os.mkdir(out_dir)
except:
    print(f"Experiment folder already exists") 
    
fleurs_hausa_train = load_dataset("google/fleurs", "ha_ng", split="train", cache_dir=cache_dir)
fleurs_hausa_val = load_dataset("google/fleurs", "ha_ng", split="validation", cache_dir=cache_dir)
fleurs_hausa_test = load_dataset("google/fleurs", "ha_ng", split="test", cache_dir=cache_dir)


# In[7]:


feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_card)


# In[9]:


tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_card, language="Hausa", task="transcribe")


# In[22]:


"""input_str = fleurs_hausa_train[0]["raw_transcription"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")"""


# In[20]:


"""tokenizer.get_vocab()"""


# In[28]:


from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(pretrained_model_card, language="Hausa", task="transcribe")


# In[31]:


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcription"]).input_ids # TODO: this or raw transcriptions?
    return batch

fleurs_hausa_train = fleurs_hausa_train.map(prepare_dataset, num_proc=4)
fleurs_hausa_val = fleurs_hausa_val.map(prepare_dataset, num_proc=4)
fleurs_hausa_test = fleurs_hausa_test.map(prepare_dataset, num_proc=4)


# In[32]:


import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

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


# In[35]:


import evaluate

metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")


# In[36]:


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


# In[ ]:


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_card)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from transformers import Seq2SeqTrainingArguments

per_device_train_batch_size = 16
learning_rate = 1e-5
num_train_epochs = 30
warmup_steps = 500
max_steps = 4000
gradient_accumulation_steps = 1
generation_max_length = 225
    

training_args = Seq2SeqTrainingArguments(
    output_dir=out_dir, 
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    warmup_steps=warmup_steps,
    max_steps=max_steps,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=generation_max_length,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=fleurs_hausa_train,
    eval_dataset=fleurs_hausa_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

import time
start = time.time()
trainer.train()
end = time.time()

print(f"Training took {end-start} seconds")

hyperparameters_file = out_dir+"hyperparameters.jsonl"
with open(hyperparameters_file, "w") as f:
    obj = {"training batch size": per_device_train_batch_size,
           "learning rate": learning_rate,
           "number of training epochs": num_train_epochs,
           "warm up steps": warmup_steps,
           "max steps": max_steps,
           "gradient accumulation steps": gradient_accumulation_steps,
           "time to train in secs": end-start,
           "generation_max_length": generation_max_length}
    json.dump(obj, f)


# In[ ]:
