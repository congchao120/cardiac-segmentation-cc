from __future__ import print_function

import numpy as np
from keras import backend as K
import tensorflow as tf
from keras import optimizers
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling3D, UpSampling3D, Dropout
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model
from metrics_common import dice_coef, dice_coef_endo, dice_coef_myo, dice_coef_rv, dice_coef_loss, dice_coef_loss_endo, dice_coef_loss_myo, dice_coef_loss_rv, dice_coef_endo_each
from layer_common import mvn3d, crop
from keras.layers import Dropout, Lambda
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization


def dice_coef_endo(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2)
    y_true_endo = y_true[:,0, ..., 2]
    y_pred_endo = y_pred[:,0, ..., 2]
    intersection = K.sum(y_true_endo * y_pred_endo, axis=axes)
    summation = K.sum(y_true_endo * y_true_endo, axis=axes) + K.sum(y_pred_endo * y_pred_endo, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)

def dice_coef_myo(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for myocardium class per batch.'''
    axes = (1, 2)
    y_true_myo = y_true[:,0, ..., 1]
    y_pred_myo = y_pred[:,0, ..., 1]
    summation_true = K.sum(y_true_myo, axis=axes)
    intersection = K.sum(y_true_myo * y_pred_myo, axis=axes)
    summation = K.sum(y_true_myo * y_true_myo, axis=axes) + K.sum(y_pred_myo * y_pred_myo, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)

def dice_coef_endo_each(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2)
    y_true_endo = y_true[:,0, ..., 2].astype('float32')
    y_pred_endo = y_pred[:,0, ..., 2]
    y_pred_endo = np.where(y_pred_endo > 0.5, 1.0, 0.0).astype('float32')
    intersection = np.sum(y_true_endo * y_pred_endo, axis=axes)
    summation = np.sum(y_true_endo * y_true_endo, axis=axes) + np.sum(y_pred_endo * y_pred_endo, axis=axes)

    return (2.0 * intersection + smooth) / (summation + smooth)

def dice_coef_myo_each(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2)
    y_true_myo = y_true[:,0, ...,  1].astype('float32')
    y_pred_myo = y_pred[:,0, ...,  1]
    y_pred_myo = np.where(y_pred_myo > 0.5, 1.0, 0.0).astype('float32')
    intersection = np.sum(y_true_myo * y_pred_myo, axis=axes)
    summation = np.sum(y_true_myo * y_true_myo, axis=axes) + np.sum(y_pred_myo * y_pred_myo, axis=axes)

    return (2.0 * intersection + smooth) / (summation + smooth)

def dice_coef_epi(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for myocardium class per batch.'''
    axes = (1, 2)
    y_true_myo = y_true[:, 0, ...,  1]
    y_pred_myo = y_pred[:, 0, ...,  1]
    y_true_endo = y_true[:, 0, ...,  2]
    y_pred_endo = y_pred[:, 0, ...,  2]

    y_true_epi = tf.cast(tf.logical_or(tf.cast(y_true_myo, tf.bool), tf.cast(y_true_endo, tf.bool)), tf.float32)
    y_pred_epi = tf.cast(tf.logical_or(tf.cast(y_pred_myo, tf.bool), tf.cast(y_pred_endo, tf.bool)), tf.float32)
    tf.summary.image("y_true_myo", y_true_myo[...,None], max_outputs=1)
    tf.summary.image("y_true_endo", y_true_endo[...,None], max_outputs=1)
    tf.summary.image("y_pred_myo", y_pred_myo[...,None], max_outputs=1)
    tf.summary.image("y_pred_endo", y_pred_endo[..., None], max_outputs=1)
    tf.summary.image("y_pred_epi", y_pred_epi[...,None], max_outputs=1)
    tf.summary.image("y_true_epi", y_true_epi[...,None], max_outputs=1)
    intersection = K.sum(y_true_epi * y_pred_epi, axis=axes)
    summation = K.sum(y_true_epi * y_true_epi, axis=axes) + K.sum(y_pred_epi * y_pred_epi, axis=axes)

    tf.summary.merge_all()
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)

def unet_lstm_multi_model(input_shape, num_classes, transfer=True, contour_type='a', weights=None):

    loss = 'categorical_crossentropy'
    activation = 'softmax'

    kwargs = dict(
        kernel_size=(3, 3),
        strides=1,
        padding='same',
        use_bias=True,
        return_sequences = True,
        trainable=True,
    )

    data = Input(shape=input_shape, dtype='float', name='data')
    mvn1 = Lambda(mvn3d, name='mvn1')(data)

    conv1 =  ConvLSTM2D(filters=32, **kwargs)(mvn1)
    #conv1 = ConvLSTM2D(filters=32, **kwargs)(conv1)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2))(conv1)
    #pool1 = Dropout(rate=0.5)(pool1)

    pool1 = Lambda(mvn3d)(pool1)
    conv2 = ConvLSTM2D(filters=64, **kwargs)(pool1)
    #conv2 = ConvLSTM2D(filters=64, **kwargs)(conv2)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
    #pool2 = Dropout(rate=0.3)(pool2)

    pool2 = Lambda(mvn3d)(pool2)
    conv3 = ConvLSTM2D(filters=128, **kwargs)(pool2)
    #conv3 = ConvLSTM2D(filters=128, **kwargs)(conv3)
    pool3 = MaxPooling3D(pool_size=(1, 2, 2))(conv3)
    #pool3 = Dropout(rate=0.5)(pool3)

    pool3 = Lambda(mvn3d)(pool3)
    conv4 = ConvLSTM2D(filters=256, **kwargs)(pool3)
    #conv4 = ConvLSTM2D(filters=256, **kwargs)(conv4)

    '''
    pool4 = MaxPooling3D(pool_size=(1, 2, 2))(conv4)
    #pool4 = Dropout(rate=0.3)(pool4)

    pool4 = Lambda(mvn3d)(pool4)
    conv5 = ConvLSTM2D(filters=512, **kwargs)(pool4)
    conv5 = ConvLSTM2D(filters=512, **kwargs)(conv5)

    up6 = concatenate([ConvLSTM2D(filters=256, **kwargs)(UpSampling3D(size=(1, 2, 2))(conv5)), conv4], axis=4)

    conv6 = ConvLSTM2D(filters=256, **kwargs)(up6)
    conv6 = ConvLSTM2D(filters=256, **kwargs)(conv6)
    '''
    up7 = concatenate([ConvLSTM2D(filters=128, **kwargs)(UpSampling3D(size=(1, 2, 2))(conv4)), conv3], axis=4)

    conv7 = ConvLSTM2D(filters=128, **kwargs)(up7)
    #conv7 = ConvLSTM2D(filters=128, **kwargs)(conv7)

    up8 = concatenate([ConvLSTM2D(filters=64, **kwargs)(UpSampling3D(size=(1, 2, 2))(conv7)), conv2], axis=4)

    conv8 = ConvLSTM2D(filters=64, **kwargs)(up8)
    #conv8 = ConvLSTM2D(filters=64, **kwargs)(conv8)

    up9 = concatenate([ConvLSTM2D(filters=32, **kwargs)(UpSampling3D(size=(1, 2, 2))(conv8)), conv1], axis=4)
    conv9 = ConvLSTM2D(filters=32, **kwargs)(up9)
    #conv9 = ConvLSTM2D(filters=32, **kwargs)(conv9)

    conv10 = Conv3D(filters=num_classes, kernel_size=(5, 1, 1),
                         strides=1, activation=activation, padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True, name="prediction")(conv9)

    model = Model(inputs=data, outputs=conv10)
    if weights is not None:
        model.load_weights(weights)

    if contour_type == 'a':
        model.compile(optimizer='adadelta', loss=loss, metrics=[dice_coef_endo, dice_coef_myo, dice_coef_epi])
    return model




if __name__ == '__main__':
    model = unet_lstm_multi_model((5, 128, 128, 1), 3, transfer=True, weights=None)
    plot_model(model, show_shapes=True, to_file='unet_lstm_multi_model.png')
    model.summary()
