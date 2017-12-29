from __future__ import print_function

import numpy as np
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Activation
from keras.optimizers import Adam
from keras.layers.merge import concatenate, add
from keras.utils.vis_utils import plot_model
from layer_common import mvn, crop
from keras.layers import Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2)
    y_true_endo = y_true[:, :, :]
    y_pred_endo = y_pred[:, :, :]
    intersection = K.sum(y_true_endo * y_pred_endo, axis=axes)
    summation = K.sum(y_true_endo * y_true_endo, axis=axes) + K.sum(y_pred_endo * y_pred_endo, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_each(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2)
    y_true_endo = y_true[:, :, :].astype('float32')
    y_pred_endo = y_pred[:, :, :]
    y_pred_endo = np.where(y_pred_endo > 0.5, 1.0, 0.0).astype('float32')
    intersection = np.sum(y_true_endo * y_pred_endo, axis=axes)
    summation = np.sum(y_true_endo * y_true_endo, axis=axes) + np.sum(y_pred_endo * y_pred_endo, axis=axes)

    return (2.0 * intersection + smooth) / (summation + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=0.0)


kwargs = dict(
    activation=None,
    padding='same',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
)

# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, kernel_size, strides=1):
    def f(input):
        conv = Conv2D(filters=nb_filter, kernel_size=kernel_size, strides=strides, **kwargs)(input)
        norm = BatchNormalization(axis=1)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, kernel_size, strides=1):
    def f(input):
        norm = BatchNormalization(axis=1)(input)
        activation = Activation("relu")(norm)
        return Conv2D(filters=nb_filter, kernel_size=kernel_size, strides=strides, **kwargs)(activation)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, strides=1):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, strides=strides)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1)(conv_3_3)
        return _shortcut(input, residual)

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, strides=1):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, strides=strides)(input)
        residual = _bn_relu_conv(nb_filters, 3)(conv1)
        return _shortcut(input, residual)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    strides = input._keras_shape[2] / residual._keras_shape[2]
    equal_channels = residual._keras_shape[3] == input._keras_shape[3]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if strides > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual._keras_shape[3], kernel_size=1, strides=int(strides), **kwargs)(input)

    return add([shortcut, residual])


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = 1
            if i == 0 and not is_first_layer:
                init_subsample = 2
            input = block_function(nb_filters=nb_filters, strides=init_subsample)(input)
        return input

    return f


def _up_block(block, mrge, nb_filters):
    up = concatenate([Conv2D(filters=2 * nb_filters, kernel_size=2, padding='same')(UpSampling2D(size=(2, 2))(block)), mrge],
                     axis=3)
    # conv = Convolution2D(4*nb_filters, 1, 1, activation='relu', border_mode='same')(up)
    conv = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', padding='same')(up)
    conv = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', padding='same')(conv)

    # conv = Convolution2D(4*nb_filters, 1, 1, activation='relu', border_mode='same')(conv)
    # conv = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same')(conv)
    # conv = Convolution2D(nb_filters, 1, 1, activation='relu', border_mode='same')(conv)

    # conv = Convolution2D(4*nb_filters, 1, 1, activation='relu', border_mode='same')(conv)
    # conv = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same')(conv)
    # conv = Convolution2D(nb_filters, 1, 1, activation='relu', border_mode='same')(conv)

    return conv


def unet_res_model_Inv_II(input_shape, num_classes, nb_filters = 32, transfer=True, contour_type='i', weights=None):

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
    mvn1 = Lambda(mvn)(data)

    conv1 = _conv_bn_relu(nb_filter=2 ** 4 * nb_filters, kernel_size=7, strides=2)(mvn1)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

    # Build residual blocks..

    block_fn = _bottleneck
    pool1 = Lambda(mvn)(pool1)
    block1 = _residual_block(block_fn, nb_filters=2 ** 4 * nb_filters, repetations=3, is_first_layer=True)(pool1)
    block1 = Lambda(mvn)(block1)
    block1 = Dropout(rate=0.3)(block1)
    block2 = _residual_block(block_fn, nb_filters=2 ** 3 * nb_filters, repetations=4)(block1)
    block2 = Lambda(mvn)(block2)
    block2 = Dropout(rate=0.3)(block2)
    block3 = _residual_block(block_fn, nb_filters=2 ** 2 * nb_filters, repetations=6)(block2)
    block3 = Lambda(mvn)(block3)
    block3 = Dropout(rate=0.3)(block3)
    block4 = _residual_block(block_fn, nb_filters=2 ** 1 * nb_filters, repetations=3)(block3)
    block4 = Lambda(mvn)(block4)


    up5 = _up_block(block4, block3, 2 ** 2 * nb_filters)
    up6 = _up_block(up5, block2, 2 ** 3 * nb_filters)
    up7 = _up_block(up6, block1, 2 ** 4 * nb_filters)
    up8 = _up_block(up7, conv1, 2 ** 4 * nb_filters)
    up9 = UpSampling2D(size=(2, 2))(up8)

    sub_output_32s = Conv2D(filters=num_classes, kernel_size=1,
                         strides=1, activation=activation, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True, name="sub_output_1")(block4)
    sub_output_16s = Conv2D(filters=num_classes, kernel_size=1,
                         strides=1, activation=activation, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True, name="sub_output_2")(up5)
    sub_output_8s = Conv2D(filters=num_classes, kernel_size=1,
                         strides=1, activation=activation, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True, name="sub_output_3")(up6)
    sub_output_4s = Conv2D(filters=num_classes, kernel_size=1,
                         strides=1, activation=activation, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True, name="sub_output_4")(up7)

    main_output = Conv2D(filters=num_classes, kernel_size=1,
                         strides=1, activation=activation, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True, name="prediction")(up9)

    model = Model(inputs=data, outputs=[main_output, sub_output_4s, sub_output_8s, sub_output_16s, sub_output_32s])
    if weights is not None:
        model.load_weights(weights)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model




if __name__ == '__main__':
    model = unet_res_model_Inv_II((128, 128, 1), 4, nb_filters=32, transfer=True, weights=None)
    plot_model(model, show_shapes=True, to_file='unet_res_model_Inv.png')
    model.summary()
