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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import load_metric
from datasets import load_dataset


cache_dir="/data/users/kashrest/asr-experiments/data/fleurs"
fleurs_hausa = load_dataset("google/fleurs", "ha_ng", cache_dir=cache_dir)
#fleurs_igbo = load_dataset("google/fleurs", "ig_ng", cache_dir=cache_dir)
#fleurs_swahili = load_dataset("google/fleurs", "sw_ke", cache_dir=cache_dir)
#fleurs_yoruba = load_dataset("google/fleurs", "yo_ng", cache_dir=cache_dir)

root = "/data/users/kashrest/asr-experiments/"
pretrained_model_card = "facebook/wav2vec2-xls-r-300m"
training_experiment_number = "3"
out_dir = root+pretrained_model_card.replace("/", "_")+"/"+"experiment_"+training_experiment_number+"/"
try:
    os.mkdir(out_dir)
except:
    print(f"Experiment folder already exists")
    

fleurs_hausa_train = load_dataset("google/fleurs", "ha_ng", split="train", cache_dir=cache_dir)
fleurs_hausa_val = load_dataset("google/fleurs", "ha_ng", split="validation", cache_dir=cache_dir)
fleurs_hausa_test = load_dataset("google/fleurs", "ha_ng", split="test", cache_dir=cache_dir)

def extract_all_chars(batch):
  all_text = " ".join(batch["transcription"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = fleurs_hausa_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=fleurs_hausa_train.column_names)
vocab_val = fleurs_hausa_val.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=fleurs_hausa_val.column_names)
vocab_test = fleurs_hausa_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=fleurs_hausa_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]) | set(vocab_val["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}

# for word delimiter, change " " --> "|" (ex. "Hello my name is Kaleen" --> "Hello|my|name|is|Kaleen")
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict) # this is for CTC to predict the end of a character (e.g. "hhh[PAD]iiiiii[PAD]" == "hi")
print(len(vocab_dict))

# Save vocabulary file
hausa_vocab_file = out_dir+"vocab_hausa_fleurs_train_val_test.json"
with open(hausa_vocab_file, 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer(hausa_vocab_file, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

sampling_rate = 16000
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=sampling_rate, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch

fleurs_hausa_train = fleurs_hausa_train.map(prepare_dataset, remove_columns=fleurs_hausa_train.column_names, num_proc=4)
fleurs_hausa_val = fleurs_hausa_val.map(prepare_dataset, remove_columns=fleurs_hausa_val.column_names, num_proc=4)
fleurs_hausa_test = fleurs_hausa_test.map(prepare_dataset, remove_columns=fleurs_hausa_test.column_names, num_proc=4)

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


import time
# https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2Config.attention_dropout
batch_size = 4
learning_rate = 3e-5
num_train_epochs = 30
attention_dropout = 0.1
hidden_dropout = 0.1
feat_proj_dropout = 0.0
mask_time_prob = 0.05
layerdrop = 0.1
warmup_steps = 500
    
hyperparameters_file = out_dir+"hyperparameters.jsonl"
with open(hyperparameters_file, "w") as f:
    obj = {"training batch size": batch_size,
           "learning rate": learning_rate,
           "number of training epochs": num_train_epochs,
           "attention dropout probability": attention_dropout,
           "hidden layer dropout probability": hidden_dropout,
           "feature projection layer dropout probability": feat_proj_dropout,
           "mask time probability": mask_time_prob,
           "layer dropout probability": layerdrop}
    json.dump(obj, f)

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
""",
    ignore_mismatched_sizes=True"""
model.num_parameters() # facebook/wav2vec2-xls-r-300m
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
  learning_rate=learning_rate,
  warmup_steps=warmup_steps,
  save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=fleurs_hausa_train,
    eval_dataset=fleurs_hausa_val, 
    tokenizer=processor.feature_extractor,
)
t1 = time.time()
trainer.train()
t2 = time.time()
print(f"Training took {t2-t1} seconds")

print(trainer.state.log_history)
