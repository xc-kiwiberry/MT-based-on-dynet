#!/bin/bash
./mt_train \
 --dynet-devices GPU:6 \
 --dynet-mem 1024 \
 --name big \
 -t ~/data_set/train/train.zh \
 -tl ~/data_set/train/train.en \
 -d ~/data_set/dev_test/nist02/nist02.zh \
 -dl ~/data_set/dev_test/nist02/nist02.en \
 --input_size 620  --hidden_size 1000  --batch_size 80  \
 --num_layers 1 \
 --save_freq 2000 --print_freq 500 \ 
