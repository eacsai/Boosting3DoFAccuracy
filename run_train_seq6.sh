#!/bin/bash

python train_kitti_2DoF_seq.py \
  --batch_size 4 \
  --gpu_id 4 \
  --proj geo \
  --name seq6 \
  --sequence 6
