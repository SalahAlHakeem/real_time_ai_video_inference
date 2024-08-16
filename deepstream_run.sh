#!/bin/bash

# Cleaning and compiling Yolov8 module
cd yolov8/nvdsinfer_custom_impl_Yolo
sudo CUDA_VER=11.4 make clean
sudo CUDA_VER=11.4 make all

# Cleaning and compiling main module
cd ../..
sudo make clean
sudo make

# Running application
sudo ./main
