#!/bin/bash
mkdir deen
cp mt_gpu.exe deen/
deen/mt_gpu.exe \
 --dynet-devices GPU:6 \
 --dynet-mem 1024 \
 --name deen \
 -t ~/data_set/wmt2017/train/deen/training.de \
 -tl ~/data_set/wmt2017/train/deen/training.en \
 -d ~/data_set/wmt2017/dev/deen/dev_deen.de \
 -dl ~/data_set/wmt2017/dev/deen/dev_deen.en \
 --input_size 620  --hidden_size 1000  --batch_size 80  \
 --num_layers 1 \
 --save_freq 2000 --print_freq 500 \ 
