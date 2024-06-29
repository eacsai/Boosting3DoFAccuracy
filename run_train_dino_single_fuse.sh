#!/bin/bash

python train_kitti_2DoF_dino.py \
  --name dino_single_fuse \
  --proj geo \
  --batch_size 24 \