#!/bin/bash

# Cleaning and compiling Yolov8 module
cd yolov8/nvdsinfer_custom_impl_Yolo
sudo CUDA_VER=11.4 make clean
sudo CUDA_VER=11.4 make all

# Cleaning and compiling main module
cd ../..
sudo make clean
sudo make

# For building TensorRT engine in jetson devices
/usr/src/tensorrt/bin/trtexec --onnx=./yolov8/yolov8s.onnx --saveEngine=./model_b4_gpu0_fp32.engine --workspace=2500 --optShapes=input:4x3x640x640

# Running application
sudo ./main
