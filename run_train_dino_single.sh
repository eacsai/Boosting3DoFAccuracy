#!/bin/bash

python train_kitti_2DoF_dino.py \
  --name dino_single_vit16b_128 \
  --proj geo \
  --batch_size 48 \