# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape, UpSampling2D, LeakyReLU, Activation, Conv2DTranspose

from keras.initializers import RandomNormal as RN, Constant
from keras.models import load_model



def D1_model(Height, Width, channel):
    inputs = Input((Height, Width, channel))
   
    x = Conv2D(128, (4, 4), padding='same', strides=(2,2), name='d_conv1',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (4, 4), padding='same', strides=(2,2), name='d_conv2',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, (4, 4), padding='same', strides=(2,2), name='d_conv3',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(1024, (4, 4), padding='same', strides=(2,2), name='d_conv4',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
              
    x = Flatten()(x)
    
    x = Dense(1, activation='sigmoid', name='d_out',
        kernel_initializer=RN(mean=0.0, stddev=0.02), bias_initializer=Constant())(x)
    model = Model(inputs=inputs, outputs=x, name='D')
    return model

g = load_model('models/G1.h5')
d = load_model('models/D1.h5')

def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model

def Combined_model1(g, d1):
    model = Sequential()
    model.add(g)
    model.add(d1)
    return model


