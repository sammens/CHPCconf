from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from losses import categorical_focal_loss
from keras.applications import InceptionV3

def build_model(act, weight, size, loss=None):
    _model = InceptionV3(
        weights=weight, # inception weights
        include_top=False,
        input_shape=(size, size, 3))
    
    GAP_layer = layers.GlobalAveragePooling2D()
    drop_layer = layers.Dropout(0.5)
    dense_layer = layers.Dense(5, activation=act, name='final_output') #args.act = sigmoid
    
    x = GAP_layer(_model.layers[-1].output)
    x = drop_layer(x)
    final_output = dense_layer(x)
    
    model = Model(_model.layers[0].input, final_output)
    if loss == loss:
        model.compile(loss=categorical_focal_loss(gamma=2.0, alpha=0.25), 
                      optimizer=Adam(lr=0.00005), 
                      metrics=["accuracy"])
    else:
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=0.00005),
            metrics=['accuracy']
        )
    
    return model