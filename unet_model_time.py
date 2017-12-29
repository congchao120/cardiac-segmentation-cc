from __future__ import print_function

import numpy as np
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Activation
from keras.optimizers import Adam
from keras.layers.merge import concatenate, add
from keras.utils.vis_utils import plot_model
from layer_common import mvn, crop
from keras.layers import Dropout, Lambda, Conv3D, merge
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (2, 3)
    y_true_endo = y_true
    y_pred_endo = y_pred
    intersection = K.sum(y_true_endo * y_pred_endo, axis=axes)
    summation = K.sum(y_true_endo * y_true_endo, axis=axes) + K.sum(y_pred_endo * y_pred_endo, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=0.0)

def unet_res_model_time(input_shape, num_classes, nb_filters = 32, n_phases=9, dilation=1, transfer=True, contour_type='i', weights=None):

    if num_classes == 2:
        num_classes = 1
        loss = dice_coef_loss
        activation = 'sigmoid'
    else:
        if transfer == True:
            if contour_type == 'i':
                loss = dice_coef_loss
            elif contour_type == 'm':
                loss = dice_coef_loss
            elif contour_type == 'r':
                loss = dice_coef_loss
            elif contour_type == 'a':
                loss = dice_coef_loss
        else:
            loss = dice_coef_loss

        activation = 'softmax'

    data = Input(shape=input_shape, dtype='float', name='data')

    conv3d_1 = Conv3D(nb_filters, kernel_size=(n_phases, 3, 3), dilation_rate=dilation, padding='same')(data)
    conv3d_1 = Activation('relu')(conv3d_1)
    conv3d_2 = Conv3D(nb_filters, kernel_size=(n_phases, 3, 3), dilation_rate=dilation, padding='same')(conv3d_1)
    conv3d_2 = Activation('relu')(conv3d_2)

    conv3d_3 = Conv3D(nb_filters, kernel_size=(n_phases, 3, 3), dilation_rate=dilation, padding='same')(conv3d_2)
    conv3d_3 = Activation('relu')(conv3d_3)
    conv3d_4 = Conv3D(nb_filters, kernel_size=(n_phases, 3, 3), dilation_rate=dilation, padding='same')(conv3d_3)
    conv3d_4 = Activation('relu')(conv3d_4)

    conv3d_5 = Conv3D(nb_filters, kernel_size=(n_phases, 3, 3), dilation_rate=dilation, padding='same')(conv3d_4)
    conv3d_5 = Activation('relu')(conv3d_5)
    conv3d_6 = Conv3D(nb_filters, kernel_size=(n_phases, 3, 3), dilation_rate=dilation, padding='same')(conv3d_5)
    conv3d_6 = Activation('relu')(conv3d_6)

    conv3d_7 = Conv3D(nb_filters, kernel_size=(n_phases, 3, 3), dilation_rate=dilation, padding='same')(conv3d_6)
    conv3d_7 = Activation('relu')(conv3d_7)
    conv3d_8 = Conv3D(nb_filters, kernel_size=(n_phases, 3, 3), dilation_rate=dilation, padding='same')(conv3d_7)
    conv3d_8 = Activation('relu')(conv3d_8)

    conv3d_9 = Conv3D(nb_filters, kernel_size=(n_phases, 3, 3), dilation_rate=dilation, padding='same')(conv3d_8)
    conv3d_9 = Activation('relu')(conv3d_9)
    conv3d_10 = Conv3D(nb_filters, kernel_size=(n_phases, 3, 3), dilation_rate=dilation, padding='same')(conv3d_9)
    conv3d_10 = Activation('relu')(conv3d_10)

    final_convolution = Conv3D(num_classes, kernel_size=(n_phases, 1, 1))(conv3d_10)
    act = Activation(activation)(final_convolution)

    model = Model(inputs=data, outputs=act)
    if weights is not None:
        model.load_weights(weights)

    model.compile(optimizer=Adam(lr=1e-6), loss=dice_coef_loss, metrics=[dice_coef])

    return model




if __name__ == '__main__':
    model = unet_res_model_time((9, 128, 128, 1), 2, nb_filters=64, n_phases=9, dilation=1, transfer=True, weights=None)
    plot_model(model, show_shapes=True, to_file='unet_res_model_time.png')
    model.summary()
