#!/bin/bash

python wav2vec2-finetuning-hausa.py "facebook/mms-1b-all" "hausa-fleurs-only-1" 16 3e-4 20 0.1 0.1 0.0 0.05 0.1 500 fleurs

python wav2vec2-finetuning-hausa.py "facebook/wav2vec2-xls-r-1b" "hausa-fleurs-only-hyperparameter-1" 16 2e-4 20 0.1 0.1 0.0 0.05 0.1 500 fleurs

python wav2vec2-finetuning-hausa.py "facebook/mms-1b" "hausa-combined-2" 16 3e-4 20 0.1 0.1 0.0 0.05 0.1 500 combined 

python wav2vec2-finetuning-hausa.py "facebook/mms-1b-all" "hausa-combined-1" 16 3e-4 3 0.1 0.1 0.0 0.05 0.1 500 combined 

python wav2vec2-finetuning-hausa.py "facebook/wav2vec2-xls-r-1b" "hausa-combined-hyperparameter-1" 16 2e-4 3 0.1 0.1 0.0 0.05 0.1 500 combined