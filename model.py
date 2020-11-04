import sys
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from losses import categorical_focal_loss

def build_model(model_name, act, weight, size, loss=None):
    if model_name == "inception":
        from keras.applications import InceptionV3
        _model = InceptionV3(
            weights=weight, # inception weights
            include_top=False,
            input_shape=(size, size, 3))
        
    elif model_name == "resnet":
        from keras.applications import ResNet50
        _model = ResNet50(
            weights=weight, # resnet weights
            include_top=False,
            input_shape=(size, size, 3))
    
    elif model_name == "VGG":
        from keras.applications import VGG16
        _model = VGG16(
            weights=weight, # VGG weights
            include_top=False,
            input_shape=(size, size, 3))
    
    elif model_name == "inceptionresnet":
        from keras.applications import InceptionResNetV2
        _model = InceptionResNetV2(
            weights=weight, # VGG weights
            include_top=False,
            input_shape=(size, size, 3))
        
    else:
        print("This model is not available.")
        sys.exit()
        
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