# -*- coding: utf-8 -*-


import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape, UpSampling2D, LeakyReLU, Activation, Conv2DTranspose

from keras.initializers import RandomNormal as RN, Constant
from keras.models import load_model


def G_model(Height, Width, channel):
    inputs = Input((100,))
    in_h = int(Height / 16)
    in_w = int(Width / 16)
    d_dim = 1024
    x = Dense(in_h * in_w * d_dim, name='g_dense1',
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_dense1_bn')(x)
    x = Reshape((in_h, in_w, d_dim), input_shape=(d_dim * in_h * in_w,))(x)

    x = Conv2DTranspose(512, (4, 4), name='g_conv1', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv1_bn')(x)

    x = Conv2DTranspose(256, (4, 4), name='g_conv2', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv2_bn')(x)

    x = Conv2DTranspose(128, (4, 4), name='g_conv3', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02), use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='g_conv3_bn')(x)
    
    x = Conv2DTranspose(channel, (4, 4), name='g_out', padding='same', strides=(2,2),
        kernel_initializer=RN(mean=0.0, stddev=0.02),  bias_initializer=Constant())(x)
    x = Activation('tanh')(x)
    model = Model(inputs=inputs, outputs=x, name='G')
    return model


def D_model(Height, Width, channel):
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


def Combined_model(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model


