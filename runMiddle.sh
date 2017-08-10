#!/bin/bash
./mt_gpu.exe \
 --dynet-mem 1024 \
 --name mid \
 -t ~/data_set/dev_test/nist03040508/nist03040508.cn \
 -tl ~/data_set/dev_test/nist03040508/nist03040508.en0 \
 -d ~/data_set/dev_test/nist06/nist06.zh \
 -dl ~/data_set/dev_test/nist06/nist06.en0 \
 --input_size 62  --hidden_size 100  --batch_size 80 \
 --num_layers 1  --num_epochs -1

