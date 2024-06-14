#!/bin/bash

python train_kitti_2DoF_seq1.py \
  --batch_size 24 \
  --gpu_id 0 \
  --proj geo \
  --name seq1_fix \
  --sequence 1 \
