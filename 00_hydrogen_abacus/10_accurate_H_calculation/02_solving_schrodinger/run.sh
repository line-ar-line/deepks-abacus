#!/bin/bash

# 编译氢原子能级求解程序
g++ -o hydrogen main.cpp hydrogen.cpp -lm

# 运行程序
./hydrogen