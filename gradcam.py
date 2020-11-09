#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:11:27 2020

@author: samuel
"""

import numpy as np
from utils import preprocess_image
from utils import load_image_ben_orig
from utils import gen_heatmap_img

NUM_SAMP=10
SEED=77
layer_name = 'mixed10' 
for i, (idx, row) in enumerate(df[:NUM_SAMP].iterrows()):
    path=f"/home/sofosumensah/lustre/PhD/data/test_resized/{row['image']}.jpeg"
    ben_img = load_image_ben_orig(path)
    input_img = np.empty((1,299, 299, 3), dtype=np.uint8)
    input_img[0,:,:,:] = preprocess_image(path)
        
    print('test pic no.%d' % (i+1))
    _ = gen_heatmap_img(input_img[0],
                        model, 
                        layer_name=layer_name, 
                        viz_img=ben_img)