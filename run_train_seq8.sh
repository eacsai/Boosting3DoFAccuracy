#!/bin/bash

python train_kitti_2DoF_seq.py \
  --batch_size 3 \
  --gpu_id 6 \
  --proj geo \
  --name seq8 \
  --sequence 8
