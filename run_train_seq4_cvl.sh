#!/bin/bash

python train_kitti_2DoF_seq1.py \
  --batch_size 32 \
  --gpu_id 0 \
  --proj geo \
  --name seq4_cvl \
  --project 'original' \
  --sequence 4
