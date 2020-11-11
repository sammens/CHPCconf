#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 02 08:52:04 2020

@author: samuel
"""

import os
import re
import cv2
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from keras import backend as K
from preprocessing import crop_clahe
 
def preprocess_image(image_path, crop, desired_size=299):
    im = Image.open(image_path)
    im = crop_clahe(im, crop)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    return im

def path(_dir, formats, data=None):
    
    base_dir = _dir
    if data == "aptos":
        df = pd.read_csv(base_dir+"Aptos/train.csv", sep=',')
    else:
        all_files = glob.glob(os.path.join(base_dir, 
                                           'train_resized', 
                                           '*.jpeg'))
        df = pd.read_csv(base_dir+"trainLabels.csv", sep=',') 
        all_files_name = [re.split(r'\/', all_files[i])[-1] for i, file in enumerate(all_files)]
        all_files_dict = {"image": all_files_name}
        pd_files = pd.DataFrame.from_dict(all_files_dict)
        pd_files['image'] = pd_files['image'].str.replace(formats, '') 
        keepImages = list(pd_files['image'])
        df = df[df['image'].isin(keepImages)]
    
    return df

def load_image_ben_orig(path, resize=True, crop=False):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0) , 10), -4, 128)
    
    return image/255

def print_pred(array_of_classes):
    xx = array_of_classes
    s1,s2 = xx.shape
    for i in range(s1):
        for j in range(s2):
            print('%.3f ' % xx[i,j],end='')
        print('')

def show_image(image, figsize=None, title=None):
    
    if figsize is not None:
        plt.figure(figsize=figsize)
        
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
        
    if title is not None:
        plt.title(title)

def show_Nimages(imgs, scale=1):

    N = len(imgs)
    fig = plt.figure(figsize=(25/scale, 16/scale))
    for i, img in enumerate(imgs):
        fig.add_subplot(1, N, i + 1, xticks=[], yticks=[])
        show_image(img)

def gen_heatmap_img(img, model0, layer_name='last_conv_layer',viz_img=None,orig_img=None):
    preds_raw = model0.predict(img[np.newaxis])
    preds = preds_raw > 0.5 
    class_idx = (preds.astype(int).sum(axis=1) - 1)[0]
    class_output_tensor = model0.output[:, class_idx]
    
    viz_layer = model0.get_layer(layer_name)
    grads = K.gradients(
                        class_output_tensor ,
                        viz_layer.output
                        )[0] 
    
    pooled_grads=K.mean(grads,axis=(0,1,2))
    iterate=K.function([model0.input],[pooled_grads, viz_layer.output[0]])
    
    pooled_grad_value, viz_layer_out_value = iterate([img[np.newaxis]])
    
    for i in range(pooled_grad_value.shape[0]):
        viz_layer_out_value[:,:,i] *= pooled_grad_value[i]
    
    heatmap = np.mean(viz_layer_out_value, axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)

    viz_img=cv2.resize(viz_img,(img.shape[1],img.shape[0]))
    heatmap=cv2.resize(heatmap,(viz_img.shape[1],viz_img.shape[0]))
    
    heatmap_color = cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_SPRING)/255
    heated_img = heatmap_color*0.5 + viz_img*0.5
    
    print('raw output from model : ')
    print_pred(preds_raw)
    
    if orig_img is None:
        show_Nimages([img,viz_img,heatmap_color,heated_img])
    else:
        show_Nimages([orig_img,img,viz_img,heatmap_color,heated_img])
    
    plt.show()
    return heated_img


