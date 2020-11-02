import os
import re
import glob

import pandas as pd

from PIL import Image
from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
 
def preprocess_image(image_path, desired_size=299):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    return im

def path(_dir, data=None):
    
    base_dir = _dir
    if data == "aptos":
        df = pd.read_csv(base_dir+"train.csv", sep=',')
    else:
        all_files = glob.glob(os.path.join(base_dir, 'train_resized', '*.jpeg'))
        df = pd.read_csv(base_dir+"trainLabels.csv", sep=',')
        all_files_name = [re.sub(r'.*/home/sofosumensah/lustre/PhD/data/train_resized/', '', i) for i in all_files]
        all_files_dict = {"image": all_files_name}
        pd_files = pd.DataFrame.from_dict(all_files_dict)
        pd_files['image'] = pd_files['image'].str.replace(r'.jpeg', '')
        keepImages = list(pd_files['image'])
        df = df[df['image'].isin(keepImages)]
    
    return df

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score( 
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model_inceptionv3_odata.h5')

        return