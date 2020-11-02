import numpy as np
import tensorflow as tf

from utils import path
from keras.backend.tensorflow_backend import set_session


config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config = config)
set_session(sess)

np.random.seed(2019)
tf.set_random_seed(2019)

train_df = path('/home/sofosumensah/lustre/PhD/data/')