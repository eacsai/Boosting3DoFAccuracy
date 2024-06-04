#!/bin/bash

python train_kitti_2DoF_seq.py \
  --batch_size 6 \
  --gpu_id 3 \
  --proj geo \
  --name seq4 \
  --sequence 4
