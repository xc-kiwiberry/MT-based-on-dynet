#!/bin/bash
g++ my_train_encdec.cc \
-o mt_gpu.exe \
-lgdynet \
-std=c++11 \
-L/global-mt/huangxuancheng/projects/dynet/build/dynet \
-L/usr/local/boost-1.60/lib -lboost_serialization-gcc49-mt-1_60

g++ my_train_encdec.cc \
-o mt_cpu.exe \
-ldynet \
-std=c++11 \
-L/global-mt/huangxuancheng/projects/dynet/build/dynet \
-L/usr/local/boost-1.60/lib -lboost_serialization-gcc49-mt-1_60

g++ my_test.cc \
-o mt_test.exe \
-lgdynet \
-std=c++11 \
-L/global-mt/huangxuancheng/projects/dynet/build/dynet \
-L/usr/local/boost-1.60/lib -lboost_serialization-gcc49-mt-1_60
