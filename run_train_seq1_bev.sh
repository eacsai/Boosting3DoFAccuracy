#!/bin/bash

python train_kitti_2DoF_seq.py \
  --batch_size 24 \
  --gpu_id 1 \
  --proj geo \
  --name seq1_bev \
  --project 'bev' \
  --sequence 1 \
