from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from losses import categorical_focal_loss
from keras.applications import InceptionV3

model = InceptionV3(
    weights='/home/sofosumensah/lustre/PhD/PyTorch_RAM/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    include_top=False,
    input_shape=(299,299,3)
)

GAP_layer = layers.GlobalAveragePooling2D()
drop_layer = layers.Dropout(0.5)
dense_layer = layers.Dense(5, activation='sigmoid', name='final_output')

def build_model(_model, loss=None):
    base_model = _model
    
    x = GAP_layer(base_model.layers[-1].output)
    x = drop_layer(x)
    final_output = dense_layer(x)
    model = Model(base_model.layers[0].input, final_output)
    if loss == 'focal_loss':
        model.compile(loss=categorical_focal_loss(gamma=2.0, alpha=0.25), optimizer=Adam(lr=0.00005), metrics=["accuracy"])
    else:
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=0.00005),
            metrics=['accuracy']
        )
    
    return model