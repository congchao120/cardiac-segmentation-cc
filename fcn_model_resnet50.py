#!/usr/bin/env python2.7

from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Lambda
from keras.layers import Input, average
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, AtrousConvolution2D
from keras.layers import ZeroPadding2D, Cropping2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras.regularizers import l2
import pylab
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from metrics_common import dice_coef, dice_coef_endo, dice_coef_myo, dice_coef_rv, dice_coef_loss, dice_coef_loss_endo, dice_coef_loss_myo, dice_coef_loss_rv, dice_coef_endo_each
from layer_common import mvn, crop


def fcn_model_resnet50(input_shape, num_classes, transfer=True, contour_type='i', weights=None):
    ''' "Skip" FCN architecture similar to Long et al., 2015
    https://arxiv.org/abs/1411.4038
    '''
    if num_classes == 2:
        num_classes = 1
        loss = dice_coef_loss
        activation = 'sigmoid'
    else:
        if transfer == True:
            if contour_type == 'i':
                loss = dice_coef_loss_endo
            elif contour_type == 'o':
                loss = dice_coef_loss_myo
            elif contour_type == 'r':
                loss = dice_coef_loss_rv
            elif contour_type == 'a':
                loss = dice_coef_loss
        else:
            loss = dice_coef_loss
        activation = 'softmax'

    kwargs_a = dict(
        kernel_size=1,
        strides=1,
        activation=None,
        padding='same',
        use_bias=False,
        kernel_initializer='glorot_uniform',
        activity_regularizer=None,
        kernel_constraint=None,
        trainable=True,
    )

    kwargs_b = dict(
        kernel_size=3,
        strides=1,
        activation=None,
        padding='same',
        use_bias=False,
        kernel_initializer='glorot_uniform',
        activity_regularizer=None,
        kernel_constraint=None,
        trainable=True,
    )

    kwargs_c = kwargs_a

    kwargs_ds = dict(
        kernel_size=1,
        strides=2,
        activation=None,
        padding='same',
        use_bias=False,
        kernel_initializer='glorot_uniform',
        activity_regularizer=None,
        kernel_constraint=None,
        trainable=True,
    )

    kwargs_atrous = dict(
        kernel_size=3,
        strides=1,
        dilation_rate=2,
        activation=None,
        padding='same',
        use_bias=False,
        kernel_initializer='glorot_uniform',
        activity_regularizer=None,
        kernel_constraint=None,
        trainable=True,
    )

    kwargs_atrous4 = dict(
        kernel_size=3,
        strides=1,
        dilation_rate=4,
        activation=None,
        padding='same',
        use_bias=False,
        kernel_initializer='glorot_uniform',
        activity_regularizer=None,
        kernel_constraint=None,
        trainable=True,
    )

    kwargs_atrous6 = dict(
        kernel_size=3,
        strides=1,
        dilation_rate=6,
        activation=None,
        padding='same',
        use_bias=False,
        kernel_initializer='glorot_uniform',
        activity_regularizer=None,
        kernel_constraint=None,
        trainable=True,
    )

    kwargs_atrous12 = dict(
        kernel_size=3,
        strides=1,
        dilation_rate=12,
        activation=None,
        padding='same',
        use_bias=False,
        kernel_initializer='glorot_uniform',
        activity_regularizer=None,
        kernel_constraint=None,
        trainable=True,
    )

    kwargs_atrous18 = dict(
        kernel_size=3,
        strides=1,
        dilation_rate=18,
        activation=None,
        padding='same',
        use_bias=False,
        kernel_initializer='glorot_uniform',
        activity_regularizer=None,
        kernel_constraint=None,
        trainable=True,
    )

    kwargs_atrous24 = dict(
        kernel_size=3,
        strides=1,
        dilation_rate=24,
        activation=None,
        padding='same',
        use_bias=False,
        kernel_initializer='glorot_uniform',
        activity_regularizer=None,
        kernel_constraint=None,
        trainable=True,
    )
    weight_decay = 1E-4

    data = Input(shape=input_shape, dtype='float', name='data')
    mvn0 = Lambda(mvn, name='mvn0')(data)
    conv1 = Conv2D(filters=64, name='conv1', kernel_size=7, strides=2, activation=None, padding='same',
                   use_bias=False, kernel_initializer='glorot_uniform')(mvn0)
    mvn1 = Lambda(mvn, name='mvn1')(conv1)
    bn1 = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), trainable=True)(mvn1)
    ac1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=3, strides=2,
                    padding='same', name='pool1')(ac1)
    #2a
    conv2a_1 = Conv2D(filters=256, name='conv2a_1', **kwargs_a)(pool1)
    mvn2a_1 = Lambda(mvn, name='mvn2a_1')(conv2a_1)
    bn2a_1 = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), trainable=True, name="bn2a_1")(mvn2a_1)

    conv2a_2a = Conv2D(filters=64, name='conv2a_2a', **kwargs_a)(pool1)
    mvn2a_2a = Lambda(mvn, name='mvn2a_2a')(conv2a_2a)
    bn2a_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), trainable=True, name="bn2a_2a")(mvn2a_2a)
    ac2a_2a = Activation('relu', name="ac2a_2a")(bn2a_2a)

    conv2a_2b = Conv2D(filters=64, name='conv2a_2b', **kwargs_b)(ac2a_2a)
    mvn2a_2b = Lambda(mvn, name='mvn2a_2b')(conv2a_2b)
    bn2a_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), trainable=True, name="bn2a_2b")(mvn2a_2b)
    ac2a_2b = Activation('relu', name="ac2a_2b")(bn2a_2b)

    conv2a_2c = Conv2D(filters=256, name='conv2a_2c', **kwargs_c)(ac2a_2b)
    mvn2a_2c = Lambda(mvn, name='mvn2a_2c')(conv2a_2c)
    bn2a_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), trainable=True, name="bn2a_2c")(mvn2a_2c)

    res2a = average([bn2a_1, bn2a_2c], name='res2a')
    ac2a= Activation('relu', name="ac2a")(res2a)

    # 2b
    conv2b_2a = Conv2D(filters=64, name='conv2b_2a', **kwargs_a)(ac2a)
    mvn2b_2a = Lambda(mvn, name='mvn2b_2a')(conv2b_2a)
    bn2b_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), trainable=True, name="bn2b_2a")(mvn2b_2a)
    ac2b_2a = Activation('relu', name="ac2b_2a")(bn2b_2a)

    conv2b_2b = Conv2D(filters=64, name='conv2b_2b', **kwargs_b)(ac2b_2a)
    mvn2b_2b = Lambda(mvn, name='mvn2b_2b')(conv2b_2b)
    bn2b_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), trainable=True, name="bn2b_2b")(mvn2b_2b)
    ac2b_2b = Activation('relu', name="ac2b_2b")(bn2b_2b)

    conv2b_2c = Conv2D(filters=256, name='conv2b_2c', **kwargs_c)(ac2b_2b)
    mvn2b_2c = Lambda(mvn, name='mvn2b_2c')(conv2b_2c)
    bn2b_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), trainable=True, name="bn2b_2c")(mvn2b_2c)

    res2b = average([ac2a, bn2b_2c], name='res2b')
    ac2b= Activation('relu', name="ac2b")(res2b)

    # 2c
    conv2c_2a = Conv2D(filters=64, name='conv2c_2a', **kwargs_a)(ac2b)
    mvn2c_2a = Lambda(mvn, name='mvn2c_2a')(conv2c_2a)
    bn2c_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn2c_2a")(mvn2c_2a)
    ac2c_2a = Activation('relu', name="ac2c_2a")(bn2c_2a)

    conv2c_2b = Conv2D(filters=64, name='conv2c_2b', **kwargs_b)(ac2c_2a)
    mvn2c_2b = Lambda(mvn, name='mvn2c_2b')(conv2c_2b)
    bn2c_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn2c_2b")(mvn2c_2b)
    ac2c_2b = Activation('relu', name="ac2c_2b")(bn2c_2b)

    conv2c_2c = Conv2D(filters=256, name='conv2c_2c', **kwargs_c)(ac2c_2b)
    mvn2c_2c = Lambda(mvn, name='mvn2c_2c')(conv2c_2c)
    bn2c_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn2c_2c")(mvn2c_2c)

    res2c = average([ac2b, bn2c_2c], name='res2c')
    ac2c = Activation('relu', name="ac2c")(res2c)
    drop2c = Dropout(rate=0.5, name='drop2c')(ac2c)

    # 3a
    conv3a_1 = Conv2D(filters=512, name='conv3a_1', **kwargs_ds)(drop2c)
    mvn3a_1 = Lambda(mvn, name='mvn3a_1')(conv3a_1)
    bn3a_1 = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                trainable=True, name="bn3a_1")(mvn3a_1)

    conv3a_2a = Conv2D(filters=128, name='conv3a_2a', **kwargs_ds)(drop2c)
    mvn3a_2a = Lambda(mvn, name='mvn3a_2a')(conv3a_2a)
    bn3a_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn3a_2a")(mvn3a_2a)
    ac3a_2a = Activation('relu', name="ac3a_2a")(bn3a_2a)

    conv3a_2b = Conv2D(filters=128, name='conv3a_2b', **kwargs_b)(ac3a_2a)
    mvn3a_2b = Lambda(mvn, name='mvn3a_2b')(conv3a_2b)
    bn3a_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn3a_2b")(mvn3a_2b)
    ac3a_2b = Activation('relu', name="ac3a_2b")(bn3a_2b)

    conv3a_2c = Conv2D(filters=512, name='conv3a_2c', **kwargs_c)(ac3a_2b)
    mvn3a_2c = Lambda(mvn, name='mvn3a_2c')(conv3a_2c)
    bn3a_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn3a_2c")(mvn3a_2c)

    res3a = average([bn3a_1, bn3a_2c], name='res3a')
    ac3a = Activation('relu', name="ac3a")(res3a)

    # 3b1
    conv3b1_2a = Conv2D(filters=128, name='conv3b1_2a', **kwargs_a)(ac3a)
    mvn3b1_2a = Lambda(mvn, name='mvn3b1_2a')(conv3b1_2a)
    bn3b1_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn3b1_2a")(mvn3b1_2a)
    ac3b1_2a = Activation('relu', name="ac3b1_2a")(bn3b1_2a)

    conv3b1_2b = Conv2D(filters=128, name='conv3b1_2b', **kwargs_b)(ac3b1_2a)
    mvn3b1_2b = Lambda(mvn, name='mvn3b1_2b')(conv3b1_2b)
    bn3b1_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn3b1_2b")(mvn3b1_2b)
    ac3b1_2b = Activation('relu', name="ac3b1_2b")(bn3b1_2b)

    conv3b1_2c = Conv2D(filters=512, name='conv3b1_2c', **kwargs_c)(ac3b1_2b)
    bn3b1_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn3b1_2c")(conv3b1_2c)

    res3b1 = average([ac3a, bn3b1_2c], name='res3b1')
    ac3b1 = Activation('relu', name="ac3b1")(res3b1)

    # 3b2
    conv3b2_2a = Conv2D(filters=128, name='conv3b2_2a', **kwargs_a)(ac3b1)
    mvn3b2_2a = Lambda(mvn, name='mvn3b2_2a')(conv3b2_2a)
    bn3b2_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn3b2_2a")(mvn3b2_2a)
    ac3b2_2a = Activation('relu', name="ac3b2_2a")(bn3b2_2a)

    conv3b2_2b = Conv2D(filters=128, name='conv3b2_2b', **kwargs_b)(ac3b2_2a)
    mvn3b2_2b = Lambda(mvn, name='mvn3b2_2b')(conv3b2_2b)
    bn3b2_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn3b2_2b")(mvn3b2_2b)
    ac3b2_2b = Activation('relu', name="ac3b2_2b")(bn3b2_2b)

    conv3b2_2c = Conv2D(filters=512, name='conv3b2_2c', **kwargs_c)(ac3b2_2b)
    bn3b2_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn3b2_2c")(conv3b2_2c)

    res3b2 = average([ac3b1, bn3b2_2c], name='res3b2')
    ac3b2 = Activation('relu', name="ac3b2")(res3b2)

    # 3b3
    conv3b3_2a = Conv2D(filters=128, name='conv3b3_2a', **kwargs_a)(ac3b2)
    mvn3b3_2a = Lambda(mvn, name='mvn3b3_2a')(conv3b3_2a)
    bn3b3_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn3b3_2a")(mvn3b3_2a)
    ac3b3_2a = Activation('relu', name="ac3b3_2a")(bn3b3_2a)

    conv3b3_2b = Conv2D(filters=128, name='conv3b3_2b', **kwargs_b)(ac3b3_2a)
    mvn3b3_2b = Lambda(mvn, name='mvn3b3_2b')(conv3b3_2b)
    bn3b3_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn3b3_2b")(mvn3b3_2b)
    ac3b3_2b = Activation('relu', name="ac3b3_2b")(bn3b3_2b)

    conv3b3_2c = Conv2D(filters=512, name='conv3b3_2c', **kwargs_c)(ac3b3_2b)
    bn3b3_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn3b3_2c")(conv3b3_2c)

    res3b3 = average([ac3b2, bn3b3_2c], name='res3b3')
    ac3b3 = Activation('relu', name="ac3b3")(res3b3)

    # 4a
    conv4a_1 = Conv2D(filters=1024, name='conv4a_1', **kwargs_a)(ac3b3) # not using down sampling, using atrous convolution layer instead
    mvn4a_1 = Lambda(mvn, name='mvn4a_1')(conv4a_1)
    bn4a_1 = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                trainable=True, name="bn4a_1")(mvn4a_1)

    conv4a_2a = Conv2D(filters=256, name='conv4a_2a', **kwargs_a)(ac3b3) # not using down sampling, using atrous convolution layer instead
    mvn4a_2a = Lambda(mvn, name='mvn4a_2a')(conv4a_2a)
    bn4a_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn4a_2a")(mvn4a_2a)
    ac4a_2a = Activation('relu', name="ac4a_2a")(bn4a_2a)

    conv4a_2b = Conv2D(filters=256, name='conv4a_2b', **kwargs_atrous)(ac4a_2a)#atrous convolution layer
    mvn4a_2b = Lambda(mvn, name='mvn4a_2b')(conv4a_2b)
    bn4a_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn4a_2b")(mvn4a_2b)
    ac4a_2b = Activation('relu', name="ac4a_2b")(bn4a_2b)

    conv4a_2c = Conv2D(filters=1024, name='conv4a_2c', **kwargs_c)(ac4a_2b)
    mvn4a_2c = Lambda(mvn, name='mvn4a_2c')(conv4a_2c)
    bn4a_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn4a_2c")(mvn4a_2c)

    res4a = average([bn4a_1, bn4a_2c], name='res4a')
    ac4a = Activation('relu', name="ac4a")(res4a)

    # 4b1
    conv4b1_2a = Conv2D(filters=256, name='conv4b1_2a', **kwargs_a)(ac4a)
    mvn4b1_2a = Lambda(mvn, name='mvn4b1_2a')(conv4b1_2a)
    bn4b1_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn4b1_2a")(mvn4b1_2a)
    ac4b1_2a = Activation('relu', name="ac4b1_2a")(bn4b1_2a)

    conv4b1_2b = Conv2D(filters=256, name='conv4b1_2b', **kwargs_atrous)(ac4b1_2a)#atrous convolution layer
    mvn4b1_2b = Lambda(mvn, name='mvn4b1_2b')(conv4b1_2b)
    bn4b1_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn4b1_2b")(mvn4b1_2b)
    ac4b1_2b = Activation('relu', name="ac4b1_2b")(bn4b1_2b)

    conv4b1_2c = Conv2D(filters=1024, name='conv4b1_2c', **kwargs_c)(ac4b1_2b)
    bn4b1_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b1_2c")(conv4b1_2c)

    res4b1 = average([ac4a, bn4b1_2c], name='res4b1')
    ac4b1 = Activation('relu', name="ac4b1")(res4b1)

    # 4b2
    conv4b2_2a = Conv2D(filters=256, name='conv4b2_2a', **kwargs_a)(ac4b1)
    mvn4b2_2a = Lambda(mvn, name='mvn4b2_2a')(conv4b2_2a)
    bn4b2_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b2_2a")(mvn4b2_2a)
    ac4b2_2a = Activation('relu', name="ac4b2_2a")(bn4b2_2a)

    conv4b2_2b = Conv2D(filters=256, name='conv4b2_2b', **kwargs_atrous)(ac4b2_2a)#atrous convolution layer
    mvn4b2_2b = Lambda(mvn, name='mvn4b2_2b')(conv4b2_2b)
    bn4b2_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b2_2b")(mvn4b2_2b)
    ac4b2_2b = Activation('relu', name="ac4b2_2b")(bn4b2_2b)

    conv4b2_2c = Conv2D(filters=1024, name='conv4b2_2c', **kwargs_c)(ac4b2_2b)
    bn4b2_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b2_2c")(conv4b2_2c)

    res4b2 = average([ac4b1, bn4b2_2c], name='res4b2')
    ac4b2 = Activation('relu', name="ac4b2")(res4b2)

    # 4b3
    conv4b3_2a = Conv2D(filters=256, name='conv4b3_2a', **kwargs_a)(ac4b2)
    mvn4b3_2a = Lambda(mvn, name='mvn4b3_2a')(conv4b3_2a)
    bn4b3_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b3_2a")(mvn4b3_2a)
    ac4b3_2a = Activation('relu', name="ac4b3_2a")(bn4b3_2a)

    conv4b3_2b = Conv2D(filters=256, name='conv4b3_2b', **kwargs_atrous)(ac4b3_2a)#atrous convolution layer
    mvn4b3_2b = Lambda(mvn, name='mvn4b3_2b')(conv4b3_2b)
    bn4b3_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b3_2b")(mvn4b3_2b)
    ac4b3_2b = Activation('relu', name="ac4b3_2b")(bn4b3_2b)

    conv4b3_2c = Conv2D(filters=1024, name='conv4b3_2c', **kwargs_c)(ac4b3_2b)
    bn4b3_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b3_2c")(conv4b3_2c)

    res4b3 = average([ac4b2, bn4b3_2c], name='res4b3')
    ac4b3 = Activation('relu', name="ac4b3")(res4b3)

    # 4b4
    conv4b4_2a = Conv2D(filters=256, name='conv4b4_2a', **kwargs_a)(ac4b3)
    mvn4b4_2a = Lambda(mvn, name='mvn4b4_2a')(conv4b4_2a)
    bn4b4_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b4_2a")(mvn4b4_2a)
    ac4b4_2a = Activation('relu', name="ac4b4_2a")(bn4b4_2a)

    conv4b4_2b = Conv2D(filters=256, name='conv4b4_2b', **kwargs_atrous)(ac4b4_2a)#atrous convolution layer
    mvn4b4_2b = Lambda(mvn, name='mvn4b4_2b')(conv4b4_2b)
    bn4b4_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b4_2b")(mvn4b4_2b)
    ac4b4_2b = Activation('relu', name="ac4b4_2b")(bn4b4_2b)

    conv4b4_2c = Conv2D(filters=1024, name='conv4b4_2c', **kwargs_c)(ac4b4_2b)
    bn4b4_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b4_2c")(conv4b4_2c)

    res4b4 = average([ac4b3, bn4b4_2c], name='res4b4')
    ac4b4 = Activation('relu', name="ac4b4")(res4b4)

    # 4b5
    conv4b5_2a = Conv2D(filters=256, name='conv4b5_2a', **kwargs_a)(ac4b4)
    mvn4b5_2a = Lambda(mvn, name='mvn4b5_2a')(conv4b5_2a)
    bn4b5_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b5_2a")(mvn4b5_2a)
    ac4b5_2a = Activation('relu', name="ac4b5_2a")(bn4b5_2a)

    conv4b5_2b = Conv2D(filters=256, name='conv4b5_2b', **kwargs_atrous)(ac4b5_2a)#atrous convolution layer
    mvn4b5_2b = Lambda(mvn, name='mvn4b5_2b')(conv4b5_2b)
    bn4b5_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b5_2b")(mvn4b5_2b)
    ac4b5_2b = Activation('relu', name="ac4b5_2b")(bn4b5_2b)

    conv4b5_2c = Conv2D(filters=1024, name='conv4b5_2c', **kwargs_c)(ac4b5_2b)
    bn4b5_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                  trainable=True, name="bn4b5_2c")(conv4b5_2c)

    res4b5 = average([ac4b4, bn4b5_2c], name='res4b5')
    ac4b5 = Activation('relu', name="ac4b5")(res4b5)

    # 5a
    conv5a_1 = Conv2D(filters=2048, name='conv5a_1', **kwargs_a)(ac4b5)#not downsampling, using atrous conv instead
    mvn5a_1 = Lambda(mvn, name='mvn5a_1')(conv5a_1)
    bn5a_1 = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                trainable=True, name="bn5a_1")(mvn5a_1)

    conv5a_2a = Conv2D(filters=512, name='conv5a_2a', **kwargs_a)(ac4b5)
    mvn5a_2a = Lambda(mvn, name='mvn5a_2a')(conv5a_2a)
    bn5a_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn5a_2a")(mvn5a_2a)
    ac5a_2a = Activation('relu', name="ac5a_2a")(bn5a_2a)

    conv5a_2b = Conv2D(filters=512, name='conv5a_2b', **kwargs_atrous4)(ac5a_2a)#atrous conv
    mvn5a_2b = Lambda(mvn, name='mvn5a_2b')(conv5a_2b)
    bn5a_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn5a_2b")(mvn5a_2b)
    ac5a_2b = Activation('relu', name="ac5a_2b")(bn5a_2b)

    conv5a_2c = Conv2D(filters=2048, name='conv5a_2c', **kwargs_c)(ac5a_2b)
    mvn5a_2c = Lambda(mvn, name='mvn5a_2c')(conv5a_2c)
    bn5a_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn5a_2c")(mvn5a_2c)

    res5a = average([bn5a_1, bn5a_2c], name='res5a')
    ac5a = Activation('relu', name="ac5a")(res5a)

    # 5b
    conv5b_2a = Conv2D(filters=512, name='conv5b_2a', **kwargs_a)(ac5a)
    mvn5b_2a = Lambda(mvn, name='mvn5b_2a')(conv5b_2a)
    bn5b_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay),
                                   beta_regularizer=l2(weight_decay),
                                   trainable=True, name="bn5b_2a")(mvn5b_2a)
    ac5b_2a = Activation('relu', name="ac5b_2a")(bn5b_2a)

    conv5b_2b = Conv2D(filters=512, name='conv5b_2b', **kwargs_atrous4)(ac5b_2a)#atrous conv
    mvn5b_2b = Lambda(mvn, name='mvn5b_2b')(conv5b_2b)
    bn5b_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay),
                                   beta_regularizer=l2(weight_decay),
                                   trainable=True, name="bn5b_2b")(mvn5b_2b)
    ac5b_2b = Activation('relu', name="ac5b_2b")(bn5b_2b)

    conv5b_2c = Conv2D(filters=2048, name='conv5b_2c', **kwargs_c)(ac5b_2b)
    bn5b_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay),
                                   beta_regularizer=l2(weight_decay),
                                   trainable=True, name="bn5b_2c")(conv5b_2c)

    res5b = average([ac5a, bn5b_2c], name='res5b')
    ac5b = Activation('relu', name="ac5b")(res5b)

    # 5c
    conv5c_2a = Conv2D(filters=512, name='conv5c_2a', **kwargs_a)(ac5b)
    mvn5c_2a = Lambda(mvn, name='mvn5c_2a')(conv5c_2a)
    bn5c_2a = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay),
                                 beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn5c_2a")(mvn5c_2a)
    ac5c_2a = Activation('relu', name="ac5c_2a")(bn5c_2a)

    conv5c_2b = Conv2D(filters=512, name='conv5c_2b', **kwargs_atrous4)(ac5c_2a)#atrous conv
    mvn5c_2b = Lambda(mvn, name='mvn5c_2b')(conv5c_2b)
    bn5c_2b = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay),
                                 beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn5c_2b")(mvn5c_2b)
    ac5c_2b = Activation('relu', name="ac5c_2b")(bn5c_2b)

    conv5c_2c = Conv2D(filters=2048, name='conv5c_2c', **kwargs_c)(ac5c_2b)
    bn5c_2c = BatchNormalization(axis=1, gamma_regularizer=l2(weight_decay),
                                 beta_regularizer=l2(weight_decay),
                                 trainable=True, name="bn5c_2c")(conv5c_2c)

    res5c = average([ac5b, bn5c_2c], name='res5c')
    ac5c = Activation('relu', name="ac5c")(res5c)
    drop5c = Dropout(rate=0.5, name='drop5c')(ac5c)

    fc1_c0 = Conv2D(filters=num_classes, name='fc1_c0', **kwargs_atrous)(drop5c)  # atrous conv
    fc1_c1 = Conv2D(filters=num_classes, name='fc1_c1', **kwargs_atrous4)(drop5c)  # atrous conv

    fc1 = average([fc1_c0, fc1_c1], name='fc1')
    us1 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                          strides=2, activation=None, padding='same',
                          kernel_initializer='glorot_uniform', use_bias=False,
                          name='us1')(fc1)

    fc2_c0 = Conv2D(filters=num_classes, name='fc2_c0', **kwargs_atrous)(drop2c)  # atrous conv
    fc2_c1 = Conv2D(filters=num_classes, name='fc2_c1', **kwargs_atrous4)(drop2c)  # atrous conv
    fc2 = average([fc2_c0, fc2_c1], name='fc2')
    crop1 = Lambda(crop, name='crop1')([fc2, us1])
    fuse1 = average([crop1, fc2], name='fuse1')

    us2 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                          strides=2, activation=None, padding='same',
                          kernel_initializer='glorot_uniform', use_bias=False,
                          name='us2')(fuse1)


    crop2 = Lambda(crop, name='crop2')([data, us2])

    fc3_c0 = Conv2D(filters=num_classes, name='fc3_c0', **kwargs_atrous)(ac1)  # atrous conv
    fc3_c1 = Conv2D(filters=num_classes, name='fc3_c1', **kwargs_atrous4)(ac1)  # atrous conv

    fc3 = average([fc3_c0, fc3_c1], name='fc3')
    crop3 = Lambda(crop, name='crop3')([fc3, us2])
    fuse2 = average([crop3, fc3], name='fuse2')

    us3 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                          strides=2, activation=None, padding='same',
                          kernel_initializer='glorot_uniform', use_bias=False,
                          name='us3')(fuse2)

    crop4 = Lambda(crop, name='crop4')([data, us3])
    predictions = Conv2D(filters=num_classes, kernel_size=1,
                         strides=1, activation=activation, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True,
                         name='predictions')(crop4)

    model = Model(inputs=data, outputs=predictions)

    if transfer == True:
        if weights is not None:
            model.load_weights(weights)
            for layer in model.layers[:10]:
                layer.trainable = False

        sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss=loss,
                      metrics=['accuracy', dice_coef_endo])
    else:
        sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss=loss,
                      metrics=['accuracy', dice_coef_endo, dice_coef_myo, dice_coef_rv])


    return model


if __name__ == '__main__':
    model = fcn_model_resnet50((100, 100, 1), 4, transfer=True, weights=None)
    plot_model(model, show_shapes=True, to_file='fcn_model_resnet50.png')
    model.summary()

