# -*- coding: utf-8 -*-


import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape, UpSampling2D, LeakyReLU, Activation, Conv2DTranspose

from keras.initializers import RandomNormal as RN, Constant
from keras.models import load_model



g = load_model('models/G1.h5')
d = load_model('models/D1.h5')

def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model

