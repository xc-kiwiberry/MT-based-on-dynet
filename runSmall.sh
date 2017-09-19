#!/bin/bash
./mt_cpu.exe \
  --dynet-devices CPU \
  --dynet-mem 1024 \
  --name toy \
  -t ~/data_set/toyData/toy.zh \
  -tl ~/data_set/toyData/toy.en \
  -d ~/data_set/toyData/valid.zh \
  -dl ~/data_set/toyData/valid.en \
  --input_size 60  --hidden_size 100  --batch_size 2 \
  --num_layers 1 \
  --print_freq 100 --save_freq 100 \
