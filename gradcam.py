#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:11:27 2020

@author: samuel
"""

import argparse

import numpy as np
# import tensorflow as tf

from keras.models import load_model
from utils import preprocess_image
from utils import load_image_ben_orig
from utils import gen_heatmap_img
# from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# sess = tf.Session(config=config)
# set_session(sess)

NUM_SAMP=10
SEED=2019
layer_name = 'mixed10' 
    
def cam(img_path, model):
    ben_img = load_image_ben_orig(img_path)
    input_img = np.empty((1, 299, 299, 3), dtype=np.uint8)
    input_img[0,:,:,:] = preprocess_image(img_path)
    return gen_heatmap_img(input_img[0],
                           model, 
                           layer_name=layer_name,
                           viz_img=ben_img)

parser = argparse.ArgumentParser('parameters')

parser.add_argument('--model', '-m', type=str, default='inception',
                    help='The type of model to use')
parser.add_argument('--path', '-p', type=str, 
                    help='The path to the image')

args = parser.parse_args()
if __name__ == '__main__':
    
    model = load_model(args.model)
    cam(args.path, model)