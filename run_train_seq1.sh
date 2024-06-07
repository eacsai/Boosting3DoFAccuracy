#!/bin/bash

python train_kitti_2DoF_seq1.py \
  --batch_size 16 \
  --gpu_id 0 \
  --proj geo \
  --name seq1 \
  --sequence 1 \
