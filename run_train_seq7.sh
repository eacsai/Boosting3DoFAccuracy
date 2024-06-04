#!/bin/bash

python train_kitti_2DoF_seq.py \
  --batch_size 4 \
  --gpu_id 5 \
  --proj geo \
  --name seq7 \
  --sequence 7
