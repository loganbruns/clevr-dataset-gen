#!/bin/bash

export CYCLES_CUDA_EXTRA_CFLAGS="-ccbin gcc-6"
export CUDA_VISIBLE_DEVICES=1

time blender --background --python render_images.py -- --num_images 150 --use_gpu 1 --split test

