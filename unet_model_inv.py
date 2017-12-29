from __future__ import print_function

import numpy as np
from keras import optimizers
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model
from layer_common import mvn, crop
from keras.layers import Dropout, Lambda
from keras import backend as K

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

def unet_model_inv(input_shape, num_classes, num_filters=32, transfer=True, contour_type='i', weights=None):

    if num_classes == 2:
        num_classes = 1
        loss = dice_coef_loss
        activation = 'sigmoid'
    else:
        if transfer == True:
            if contour_type == 'i':
                loss = dice_coef_loss
            elif contour_type == 'o':
                loss = dice_coef_loss
            elif contour_type == 'r':
                loss = dice_coef_loss
            elif contour_type == 'a':
                loss = dice_coef_loss
        else:
            loss = dice_coef_loss
        activation = 'softmax'

    kwargs = dict(
        kernel_size=3,
        strides=1,
        activation='relu',
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

    data = Input(shape=input_shape, dtype='float', name='data')
    mvn1 = Lambda(mvn, name='mvn1')(data)
    conv1 =  Conv2D(filters=2**4*num_filters, **kwargs)(mvn1)
    conv1 = Conv2D(filters=2**4*num_filters, **kwargs)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #pool1 = Dropout(rate=0.5)(pool1)

    pool1 = Lambda(mvn)(pool1)
    conv2 = Conv2D(filters=2**3*num_filters, **kwargs)(pool1)
    conv2 = Conv2D(filters=2**3*num_filters, **kwargs)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool2 = Dropout(rate=0.3)(pool2)

    pool2 = Lambda(mvn)(pool2)
    conv3 = Conv2D(filters=2**2*num_filters, **kwargs)(pool2)
    conv3 = Conv2D(filters=2**2*num_filters, **kwargs)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #pool3 = Dropout(rate=0.5)(pool3)

    pool3 = Lambda(mvn)(pool3)
    conv4 = Conv2D(filters=2**1*num_filters, **kwargs)(pool3)
    conv4 = Conv2D(filters=2**1*num_filters, **kwargs)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #pool4 = Dropout(rate=0.3)(pool4)

    pool4 = Lambda(mvn)(pool4)
    conv5 = Conv2D(filters=2**0*num_filters, **kwargs)(pool4)
    conv5 = Conv2D(filters=2**0*num_filters, **kwargs)(conv5)
    # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool5)
    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(convdeep)

    # upmid = merge([Convolution2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(convdeep)), conv5], mode='concat', concat_axis=1)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(upmid)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(convmid)

    #up6 = merge(
    #    [Conv2D(filters=256, **kwargs)(UpSampling2D(size=(2, 2))(conv5)), conv4],
    #    mode='concat', concat_axis=3)
    up6 = concatenate([Conv2D(filters=2**1*num_filters, **kwargs)(UpSampling2D(size=(2, 2))(conv5)), conv4], axis=3)
    #up6 = Lambda(mvn)(up6)
    conv6 = Conv2D(filters=2**1*num_filters, **kwargs)(up6)
    conv6 = Conv2D(filters=2**1*num_filters, **kwargs)(conv6)
    #conv6 = Dropout(rate=0.5)(conv6)

    #conv6 = Lambda(mvn)(conv6)
    #up7 = merge(
    #    [Conv2D(filters=128, **kwargs)(UpSampling2D(size=(2, 2))(conv6)), conv3],
    #    mode='concat', concat_axis=3)
    up7 = concatenate([Conv2D(filters=2**2*num_filters, **kwargs)(UpSampling2D(size=(2, 2))(conv6)), conv3], axis=3)
    #up7 = Lambda(mvn)(up7)
    conv7 = Conv2D(filters=2**2*num_filters, **kwargs)(up7)
    conv7 = Conv2D(filters=2**2*num_filters, **kwargs)(conv7)
    #conv7 = Dropout(rate=0.5)(conv7)

    #conv7 = Lambda(mvn)(conv7)
    #up8 = merge(
    #    [Conv2D(filters=64, **kwargs)(UpSampling2D(size=(2, 2))(conv7)), conv2],
    #    mode='concat', concat_axis=3)
    up8 = concatenate([Conv2D(filters=2**3*num_filters, **kwargs)(UpSampling2D(size=(2, 2))(conv7)), conv2], axis=3)
    #up8 = Lambda(mvn)(up8)
    conv8 = Conv2D(filters=2**3*num_filters, **kwargs)(up8)
    conv8 = Conv2D(filters=2**3*num_filters, **kwargs)(conv8)
    #conv8 = Dropout(rate=0.5)(conv8)

    #conv8 = Lambda(mvn)(conv8)
    #up9 = merge(
    #    [Conv2D(filters=32, **kwargs)(UpSampling2D(size=(2, 2))(conv8)), conv1],
    #    mode='concat', concat_axis=3)
    up9 = concatenate([Conv2D(filters=2**4*num_filters, **kwargs)(UpSampling2D(size=(2, 2))(conv8)), conv1], axis=3)
    conv9 = Conv2D(filters=2**4*num_filters, **kwargs)(up9)
    conv9 = Conv2D(filters=2**4*num_filters, **kwargs)(conv9)
   # conv9 = Dropout(rate=0.5)(conv9)

    #conv9 = Lambda(mvn)(conv9)

    conv10 = Conv2D(filters=num_classes, kernel_size=1,
                         strides=1, activation=activation, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True, name="prediction")(conv9)

    model = Model(inputs=data, outputs=conv10)
    if weights is not None:
        model.load_weights(weights)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    #sgd = optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
    #model.compile(optimizer=sgd, loss=dice_coef_loss_endo,
    #              metrics=[dice_coef_endo])
    return model




if __name__ == '__main__':
    model = unet_model_inv((128, 128, 1), 4, 32, transfer=True, weights=None)
    plot_model(model, show_shapes=True, to_file='unet_model_inv.png')
    model.summary()
