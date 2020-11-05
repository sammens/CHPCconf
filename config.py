#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 02 08:52:04 2020

@author: samuel
"""

import argparse

def get_args():
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--size', '-s', type=int, default=299, 
                        help='The size of the image')
    parser.add_argument('--path', '-p', type=str,
                        help='The path to the data')
    parser.add_argument('--multi_label', type=bool, default=True,
                        help='Multi-labelling classification')
    parser.add_argument('--format', '-f', type=str, default='.jpeg',
                        help='The type of format for the data')
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='The test size for validation set')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='The batch size for training data')
    parser.add_argument('--acts', '-a', type=str, default='sigmoid',
                        help='The activation function for training')
    parser.add_argument('--weight', '-w', type=str,
                        help='Pretrained weights for training')
    parser.add_argument('--data', "-d", type=str,
                        help='Which data to use')
    parser.add_argument('--model', type=str,
                        help='which model to use')
    parser.add_argument('--oversample', type=bool, default=False,
                        help='oversampling')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for the model')
    parser.add_argument('--epochs', type=int, default=15, 
                        help='Number of epochs for training the model')
    
    args = parser.parse_args()
    
    return args