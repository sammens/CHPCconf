#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:26:09 2020

@author: samuel
"""

import sys
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

parser = argparse.ArgumentParser('parameters')

parser.add_argument("--data", "-d", type=str, default='combined',
                    help="the data distribution to plot")
parser.add_argument('--title', '-t', type=str,
                    help='title of the plot')
parser.add_argument('--show', '-s', type=bool, default=False,
                    help='show title or not')
args = parser.parse_args()

if __name__ == '__main__':
    
    data_1 = format_data('data/train.csv', data='Aptos')
    data_2 = format_data('data/trainLabels.csv')
    data_3 = format_data('data/retinopathy_solution.csv', data='eye_test')
    
    if args.data == 'combined':    
        data = pd.concat([data_1, data_2, data_3])
        title = args.title
        
    elif args.data == 'aptos':
        data = data_1
        title = args.title
    
    elif args.data == 'eyepacs':
        data = pd.concat([data_2, data_3])
        title = args.title
    
    else:
        print('This data is not available')
        sys.exit()
    
    ax = sns.countplot(
        x='Level_Name',
        data=data, order=['Normal', 
                          'Mild', 
                          'Moderate', 
                          'Severe', 
                          'Proliferative'])
    ax.set(xlabel="Disease Level")
    if args.show == True:
        ax.set_title(title) 
    plt.show()