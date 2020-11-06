#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:26:09 2020

@author: samuel
"""

import pandas as pd
import seaborn as sns

def format_data(path, data=None):
    df = pd.read_csv(path)
    column_name = ['image', 'level']
    
    if data == 'Aptos':
        df.columns = column_name
    
    elif data == 'eye_test':
        df = df.drop('Usage', axis=1)
        
    else:
        pass
    
    df.loc[df['level'] == 0, 'Level_Name'] = 'Normal' 
    df.loc[df['level'] == 1, 'Level_Name'] = 'Mild'
    df.loc[df['level'] == 2, 'Level_Name'] = 'Moderate'
    df.loc[df['level'] == 3, 'Level_Name'] = 'Severe'
    df.loc[df['level'] == 4, 'Level_Name'] = 'Proliferative'
    
    return df

if __name__ == '__main__':
    
    data_1 = format_data('data/train.csv', data='Aptos')
    data_2 = format_data('data/trainLabels.csv')
    data_3 = format_data('data/retinopathy_solution.csv', data='eye_test')
    
    merge_all = pd.concat([data_1, data_2, data_3])
    
    ax = sns.countplot(x='Level_Name', data=merge_all, order=['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
    ax.set(xlabel="Disease Level")
    
    merge_eye = pd.concat([data_2, data_3])
    ax1 = sns.countplot(x='Level_Name', data=merge_eye, order=['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
    ax1.set(xlabel="Disease Level")