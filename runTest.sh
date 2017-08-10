#!/bin/bash
./mt_test.exe \
 --dynet-mem 2300 \
 --name nist06 \
 -m big_20_1_620_1000_tloss=3305.22_dloss=59.9371.params \
 -t ~/data_set/train/train.zh \
 -tl ~/data_set/train/train.en \
 -ts ~/data_set/dev_test/nist06/nist06.zh \
 -tsl ~/data_set/dev_test/nist06/nist06.en0 \
 --input_size 620 --hidden_size 1000 --num_layers 1 \
