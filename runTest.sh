#!/bin/bash
./mt_test \
 --dynet-devices GPU:7 \
 --dynet-mem 2300 \
 --name nist06 \
 -m old_models//big_13_1_620_1000_tloss=3344.99_BLEU=35.68.params \
 -t ~/data_set/train/train.zh \
 -tl ~/data_set/train/train.en \
 -ts ~/data_set/dev_test/nist06/nist06.zh \
 -tsl ~/data_set/dev_test/nist06/nist06.en \
 --input_size 620 --hidden_size 1000 --num_layers 1 \
 --debug_info 0 \
