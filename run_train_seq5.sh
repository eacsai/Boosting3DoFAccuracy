#!/bin/bash

python train_kitti_2DoF_seq.py \
  --batch_size 4 \
  --gpu_id 3 \
  --proj geo \
  --name seq5 \
  --sequence 5
