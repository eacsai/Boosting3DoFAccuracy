#!/bin/bash

python train_kitti_2DoF_seq.py \
  --batch_size 24 \
  --gpu_id 2 \
  --proj geo \
  --name seq1 \
  --sequence 1 \
