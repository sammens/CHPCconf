import numpy as np
import tensorflow as tf

from utils import path
from utils import Metrics
from model import build_model
from data_loader import load_data
from data_loader import create_datagen
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session


config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config = config)
set_session(sess)

np.random.seed(2019)
tf.set_random_seed(2019)

train_df = path('/home/sofosumensah/lustre/PhD/data/')
count_levels = np.bincount(train_df.iloc[:,1].values)
class_weights = dict(enumerate(np.sum(count_levels)/(count_levels * count_levels.size)))
x_train, y_train = load_data(train_df, multi_label=True)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.15, 
    random_state=2019
)

BATCH_SIZE = 32

data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)

model = build_model()

kappa_metrics = Metrics()

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=15,
    validation_data=(x_val, y_val),
    class_weight=class_weights,
    max_queue_size=10, 
    workers=10,
    callbacks=[kappa_metrics]
)