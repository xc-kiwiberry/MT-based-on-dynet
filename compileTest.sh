#!/bin/bash
g++ my_test.cc -o mt_test.exe -L/global-mt/huangxuancheng/projects/dynet/build/dynet \
-lgdynet -std=c++11 -L/usr/local/boost-1.60/lib -lboost_serialization-gcc49-mt-1_60


