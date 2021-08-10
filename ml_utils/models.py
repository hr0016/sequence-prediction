'''
A module for storing the models we will work with

To use this in another place, you can say
    from ml_utils import models
    
From there you can create instances of models as
    model = models.build_convnet_model(*args)
    
'''

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv1D, SeparableConv1D, MaxPooling1D



def build_convnet_model(X_in, loss='mae', n_outputs=1):
    n_steps = X_in.shape[1]
    n_features = X_in.shape[2]

    h1_inputs = Input(shape=(n_steps, n_features))
    h1_conv1 = Conv1D(filters=32, kernel_size=6, activation='relu',  kernel_initializer='glorot_uniform')(h1_inputs)
    h1_norm1 = layers.BatchNormalization()(h1_conv1)
    h1_pool1 = MaxPooling1D(pool_size=2)(h1_norm1)

    h1_conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(h1_pool1)
    h1_conv3 = Conv1D(filters=32, kernel_size=3, activation='relu')(h1_conv2)
    h1_norm2 = layers.BatchNormalization()(h1_conv3)
    h1_pool2 = MaxPooling1D(pool_size=2)(h1_norm2)

    h1_conv4 = Conv1D(filters=64, kernel_size=3, activation='relu')(h1_pool2)
    h1_conv5 = Conv1D(filters=64, kernel_size=3, activation='relu')(h1_conv4)
    h1_norm3 = layers.BatchNormalization()(h1_conv5)
    h1_pool3 = MaxPooling1D(pool_size=2)(h1_norm3)

    h1_conv6 = Conv1D(filters=64, kernel_size=3, activation='relu')(h1_pool3)
    h1_conv7 = Conv1D(filters=64, kernel_size=3, activation='relu')(h1_conv6)
    h1_norm4 = layers.BatchNormalization()(h1_conv7)
    h1_pool4 = MaxPooling1D(pool_size=2)(h1_norm4)

    flat1 = Flatten()(h1_pool4)

    dense_pre_pred = Dense(64, activation='relu')(flat1)
    
    drop1 = Dropout(0.45)(dense_pre_pred)

    dense_pred = Dense(n_outputs)(drop1)


    _model = Model(inputs=[ h1_inputs], outputs=dense_pred)

    _model.compile(optimizer='adam', loss=loss, metrics=['mape'])

    return _model


