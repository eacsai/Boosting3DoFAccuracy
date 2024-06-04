#!/bin/bash

python train_kitti_2DoF_seq.py \
  --batch_size 6 \
  --gpu_id 2 \
  --proj geo \
  --name seq3 \
  --sequence 3
