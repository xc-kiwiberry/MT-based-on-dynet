#!/bin/bash
g++ my_train_encdec.cc \
-o mt_train \
-ldynet \
-std=c++11 \
-L/data/disk2/private/hxc/dynet/build/dynet

g++ my_test.cc \
-o mt_test \
-ldynet \
-std=c++11 \
-L/data/disk2/private/hxc/dynet/build/dynet
