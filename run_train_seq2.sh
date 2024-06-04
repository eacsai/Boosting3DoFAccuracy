#!/bin/bash

python train_kitti_2DoF_seq.py \
  --batch_size 18 \
  --gpu_id 0 \
  --proj geo \
  --name seq2 \
  --sequence 2
