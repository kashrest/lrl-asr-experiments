# lrl-asr-experiments

# Automatic Speech Recognition (ASR) Tutorial: Fine-tune a pretrained, multilingual ASR model on FLEURS

## Environment 

First, clone this GitHub repository in your local machine.

Then, create a virtual environment for your project. We recommend Anaconda package manager. You can create a conda environment with all the required packages using this command:
 `conda env create -f environment.yml` which creates a conda virtual environment named "asr" containing all the necessary packages.

## Finetuning on FLEURS

(example is Hausa)

We will be ussing the FLEURS dataset (https://arxiv.org/abs/2205.12446) which has data for 102 languages. There are 3 major open-source ASR multilingual models available:
* XLS-R 
* Whisper
* MMS

(will combine scripts for Whisper and Wav2Vec2 later)  

You can create your own preprocessing script, that takes into account linguistic aspects, or use a basic (to be added from Whisper) preprocdessing script provided that removes special characters and does lowercase (credit: Whisper). Here for our example in Hausa, we have a custom preprocessing function. Let's finetune MMS-1b-all on the FLEURS Hausa dataset by running this command using the wav2vec2-finetuning-hausa.py script in this repo:
```
python wav2vec2-finetuning-hausa.py model_card output_dir_name batch_size learning_rate num_epoch attention_dropout hidden_dropout feat_proj_dropout mask_time_prob layer_dropout warmup_steps fleurs_only
```

You can choose hyperparameters and specify the output directory.
