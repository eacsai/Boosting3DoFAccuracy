#!/bin/bash

python train_kitti_2DoF_seq.py \
  --batch_size 3 \
  --gpu_id 7 \
  --proj geo \
  --name seq9 \
  --sequence 9
