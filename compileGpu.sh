#!/bin/bash
g++ my_train_encdec.cc -o mt_gpu.exe -L/global-mt/huangxuancheng/projects/dynet/build/dynet \
-lgdynet -std=c++11 -L/usr/local/boost-1.60/lib -lboost_serialization-gcc49-mt-1_60


