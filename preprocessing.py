#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:17:26 2020

@author: samuel
"""

import cv2

import numpy as np

def _prep(img):
    sz = 224
    target_radius_size = sz/2
    radius_mask_ratio=0.9
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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