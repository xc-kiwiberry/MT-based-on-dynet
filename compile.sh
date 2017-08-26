#!/bin/bash
g++ my_train_encdec.cc \
-o mt_gpu.exe \
-ldynet \
-std=c++11 \
-L/global-mt/huangxuancheng/projects/dynet/build/dynet

g++ my_train_encdec.cc \
-o mt_cpu.exe \
-ldynet \
-std=c++11 \
-L/global-mt/huangxuancheng/projects/dynet/build/dynet

g++ my_test.cc \
-o mt_test.exe \
-ldynet \
-std=c++11 \
-L/global-mt/huangxuancheng/projects/dynet/build/dynet
