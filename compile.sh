#!/bin/bash
g++ my_train_encdec.cc \
-g -std=c++11 \
-o mt_train \
-L/data/disk2/private/hxc/dynet/build/dynet -ldynet 

g++ my_test.cc \
-g -std=c++11 \
-o mt_train \
-L/data/disk2/private/hxc/dynet/build/dynet -ldynet 
