#!/bin/bash

# python wav2vec2-finetuning-hausa.py model_card output_dir_name batch_size gradient_accumulation_steps learning_rate num_epoch attention_dropout hidden_dropout feat_proj_dropout mask_time_prob layer_dropout warmup_steps fleurs_only


python wav2vec2-finetuning.py "facebook/mms-1b-all" "igbo-fleurs-only-temp" 16 2 3e-3 20 0.1 0.1 0.0 0.05 0.1 500 fleurs 

#python wav2vec2-finetuning.py "facebook/mms-1b-all" "igbo-fleurs-only-1" 16 2 3e-3 20 0.1 0.1 0.0 0.05 0.1 500 fleurs 

#python wav2vec2-finetuning.py "facebook/mms-1b-all" "igbo-fleurs-only-2" 16 2 3e-4 20 0.1 0.1 0.0 0.05 0.1 500 fleurs 

#python wav2vec2-finetuning.py "facebook/mms-1b-all" "igbo-fleurs-only-3" 16 2 3e-5 20 0.1 0.1 0.0 0.05 0.1 500 fleurs 

#python wav2vec2-finetuning.py "facebook/wav2vec2-xls-r-1b" "igbo-fleurs-only-1" 16 2 3e-4 20 0.1 0.1 0.0 0.05 0.1 500 fleurs 
