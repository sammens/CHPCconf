import numpy as np
import pandas as pd

from tqdm import tqdm
from utils import preprocess_image
from sklearn.utils import resample
from keras.preprocessing.image import ImageDataGenerator

def load_data(labels, img_path, size=229, data=None, oversample=True, multi_label=True):
        
    if oversample:
        new_df = pd.DataFrame()
        major = labels[labels.iloc[:,1] == np.argmax(np.bincount(labels.iloc[:,1]))]
        minors = np.unique(labels.iloc[:,1])[np.unique(labels.iloc[:,1]) != np.argmax(np.bincount(labels.iloc[:,1]))]
        for i in minors:
            minor = labels[labels.iloc[:,1] == i]
            n = len(major) - len(minor)
            minor_df = resample(minor, replace=True, n_samples=n+len(minor), random_state=42)
            new_df = new_df.append(minor_df)
        new_df = new_df.reset_index(drop=True)
        labels = pd.concat([major, new_df], ignore_index=True)
        labels = labels.reset_index(drop=True)
    else:
        pass
    labels = labels.reset_index(drop=True)
    N = labels.shape[0]
    # add a size to argument
    x_train = np.empty((N, size, size, 3), dtype=np.uint8)
    y_train = pd.get_dummies(labels.iloc[:,1]).values
    
    if multi_label:
        y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
        y_train_multi[:, 4] = y_train[:, 4]
        
        for i in range(3, -1, -1):
            y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])
        
        y_train = y_train_multi
    
    for i, image_id in enumerate(tqdm(labels.iloc[:,0])):
        if data == 'aptos':
            x_train[i, :, :, :] = preprocess_image(img_path+'Aptos/train_images/{}.png'.format(image_id), size) 
        else:
            try:
                x_train[i, :, :, :] = preprocess_image(img_path+'train_resized/{}.jpeg'.format(image_id), size)
            except FileNotFoundError:
                continue
                
    return x_train, y_train

def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )