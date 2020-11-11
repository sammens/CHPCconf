#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:17:26 2020

@author: samuel
"""

import cv2
import imutils

import numpy as np

from PIL import Image  
    
def crop_image_from_gray(img, tol=7, **kwargs):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """  
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol        
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2=img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3=img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop(img, **kwargs):   
    """
    Create circular crop around image centre    
    """    

    img = crop_image_from_gray(img)    
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x, y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    
    return img 

def ben_preprocessing(img):
    sz = 224
    target_radius_size = sz/2
    radius_mask_ratio=0.9
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img[img.shape[0] // 2, :, :].sum(axis=1)
    r = (x > x.mean() / 10.0).sum() / 2.0
    s = target_radius_size / r
    img = cv2.resize(img, dsize=None, fx=s, fy=s)
    img_blurred = cv2.GaussianBlur(img, (0, 0), target_radius_size / 30)
    img = cv2.addWeighted(img, 4, img_blurred, -4, 128)
    mask = np.zeros(img.shape)
    center = (img.shape[1]//2, img.shape[0]//2)
    radius = int(500 * radius_mask_ratio)
    cv2.circle(mask, center=center, radius=radius, color=(1, 1, 1), thickness=-1)
    img = img * mask + (1 - mask) * 128
    return img

def crop_image(image):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    crop = image[y:y + h, x:x + w]
    
    img  = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    
    return img

def clahe(image):
    sz = 224
    # image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    crop = image[y:y + h, x:x + w]
    
    img_start = cv2.resize(crop, (sz, sz), interpolation = cv2.INTER_AREA) 
    
    img_hsv = cv2.cvtColor(img_start, cv2.COLOR_RGB2HSV)

    #Retrieve the value/brightness componenet of image
    brightness = img_hsv[:, :, 2]

    # Apply CLAHE algorithm on brighntess component of image to adjust image contrast 
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    b_adapteq = clahe.apply(brightness)

    # Restore brighntess component subject to CLAHE to original HSV color space
    img_hsv[:,:,2] = b_adapteq

    # Convert image from HSV to RGB color space
    img_end = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    img = cv2.cvtColor(img_end, cv2.COLOR_RGB2BGR)
    return img

def crop_clahe(image, crop):
    if crop == 'circle':
        image = circle_crop(image)
    
    elif crop == 'image':
        image = crop_image(image)
    
    else:
        pass
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = img_hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    b_adapteq = clahe.apply(brightness)    
    img_hsv[:,:,2] = b_adapteq
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    img = Image.fromarray(img)
    return img
