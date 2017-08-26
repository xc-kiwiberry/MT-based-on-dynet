#!/bin/bash
./mt_test.exe \
 --dynet-devices CPU \
 --dynet-mem 1024 \
 --name toytest \
 -m toy_1_60_100.params \
 -t ~/data_set/toyData/toy.zh \
 -tl ~/data_set/toyData/toy.en \
 -ts ~/data_set/toyData/test.zh \
 -tsl ~/data_set/toyData/test.en \
 --input_size 60 --hidden_size 100 --num_layers 1 \
 --debug_info 0 \
