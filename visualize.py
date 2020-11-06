#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:26:09 2020

@author: samuel
"""

import pandas as pd

data_1 = pd.read_csv('data/train.csv')
data_2 = pd.read_csv('data/trainLabels.csv')
data_3 = pd.read_csv('data/retinopathy_solution.csv')
data_3 = data_3.drop('Usage', axis=1)
column_name = ['image', 'level']
data_1.columns = column_name
merge_all = pd.concat([data_1, data_2, data_3])
merge_all[['level']].hist(figsize=(10, 5))

merge_eye = pd.concat([data_2, data_3])
merge_eye[['level']].hist(figsize=(10, 5))
data_1[['level']].hist(figsize=(10,5))