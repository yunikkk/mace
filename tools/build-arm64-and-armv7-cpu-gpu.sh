#!/bin/bash

set -e

LIB_DIR=build/lib
INCLUDE_DIR=build/include

mkdir -p $LIB_DIR
mkdir -p $INCLUDE_DIR

# copy include headers
cp -R include/mace $INCLUDE_DIR/

# make directories
rm -rf $LIB_DIR/armeabi-v7a
mkdir -p $LIB_DIR/armeabi-v7a/cpu_gpu_dsp
mkdir -p $LIB_DIR/armeabi-v7a/cpu_gpu

rm -rf $LIB_DIR/arm64-v8a
mkdir -p $LIB_DIR/arm64-v8a/cpu_gpu_dsp
mkdir -p $LIB_DIR/arm64-v8a/cpu_gpu
mkdir -p $LIB_DIR/arm64-v8a/cpu_gpu_apu

rm -rf $LIB_DIR/linux-x86-64
mkdir -p $LIB_DIR/linux-x86-64

rm -rf $LIB_DIR/arm_linux_gnueabihf
mkdir -p $LIB_DIR/arm_linux_gnueabihf/cpu_gpu

rm -rf $LIB_DIR/aarch64_linux_gnu
mkdir -p $LIB_DIR/aarch64_linux_gnu/cpu_gpu

echo "build static lib for armeabi-v7a + cpu_gpu"
bazel build --config android --config optimization mace/libmace:libmace_static --config symbol_hidden --define neon=true --define opencl=true --define quantize=true --cpu=armeabi-v7a
cp bazel-genfiles/mace/libmace/libmace.a $LIB_DIR/armeabi-v7a/cpu_gpu/

echo "build static lib for arm64-v8a + cpu_gpu"
bazel build --config android --config optimization mace/libmace:libmace_static --config symbol_hidden --define neon=true --define opencl=true --define quantize=true --cpu=arm64-v8a
cp bazel-genfiles/mace/libmace/libmace.a $LIB_DIR/arm64-v8a/cpu_gpu/

echo "LIB PATH: $LIB_DIR"
echo "INCLUDE FILE PATH: $INCLUDE_DIR"
