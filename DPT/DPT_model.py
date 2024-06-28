#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch.nn as nn
from torch.nn.functional import interpolate as tensor_resize

# Only Needed for image pre-/post-processing
import cv2
import numpy as np

# For basic type hints
from torch import Tensor
from numpy.typing import NDArray
from .fusion_model import *
from .reassembly_model import *

# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class DPT(nn.Module):
    
    '''
    Simplified implementation of a 'Dense Prediction Transformer' model, described in:
        "Vision Transformers for Dense Prediction"
        By: René Ranftl, Alexey Bochkovskiy, Vladlen Koltun
        @ https://arxiv.org/abs/2103.13413
    
    Original implementation details come from the MiDaS project repo:
        https://github.com/isl-org/MiDaS
    '''
    
    # .................................................................................................................
    
    def __init__(self):
        '''
        Helper used to build all Depth-Anything DPT components. The arguments for this function are
        expected to come from the 'make_depthanything_dpt_from_original_state_dict' function, which
        will use arguments based on a loaded state dictionary.
        
        However, if you want to make a model without pretrained weights
        here are the following standard configs (from Depth-Anything/DinoV2):
        
        # vit-large:
            features_per_token = 1024
            num_heads = 16
            num_blocks = 24
            reassembly_features_list = [256, 512, 1024, 1024]
            base_patch_grid_hw = (37, 37)
            fusion_channels = 256
            patch_size_px = 14
        
        # vit-base
            features_per_token = 768
            num_heads = 12
            num_blocks = 12
            reassembly_features_list = [96, 192, 384, 768]
            base_patch_grid_hw = (37, 37)
            fusion_channels = 128
            patch_size_px = 14
        
        # vit-small
            features_per_token = 384
            num_heads = 6
            num_blocks = 12
            reassembly_features_list = [48, 96, 192, 384]
            base_patch_grid_hw = (37, 37)
            fusion_channels = 64
            patch_size_px = 14
        '''
        # Inherit from parent
        super().__init__()
        
        features_per_token = 768
        reassembly_features_list = [96, 192, 384, 768]
        fusion_channels = 64
        # Store models for use in forward pass
        self.reassemble = ReassembleModel(features_per_token, reassembly_features_list, fusion_channels)
        self.fusion = FusionModel(fusion_channels)
    # .................................................................................................................
    
    def forward(self, feats: list) -> Tensor:
        
        '''
        Depth prediction function. Expects an image tensor of shape BxCxHxW, with RGB ordering.
        Pixel values should have a mean near 0.0 and a standard-deviation near 1.0
        The height & width of the image need to be compatible with the patch sizing of the model.
        
        Use the 'verify_input(...)' function to test inputs if needed.
        
        Returns single channel inverse-depth 'image' of shape: BxHxW
        '''
        
        assert len(feats) == 4

        # Process patch tokens back into (multi-scale) image-like tensors
        reasm_1, reasm_2, reasm_3, reasm_4 = self.reassemble(feats[0], feats[1], feats[2], feats[3])
        
        # Generate a single (fused) feature map from multi-stage input & project into (1ch) depth image output
        fused_feature_map = self.fusion(reasm_1, reasm_2, reasm_3, reasm_4)
        
        return fused_feature_map