#!/bin/bash

python train_kitti_2DoF_seq.py \
  --batch_size 6 \
  --gpu_id 0 \
  --proj geo \
  --name seq4_bev \
  --project 'bev' \
  --sequence 4
