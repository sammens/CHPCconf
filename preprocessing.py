#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:17:26 2020

@author: samuel
"""

import cv2
import imutils

import numpy as np

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