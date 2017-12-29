#!/usr/bin/env python2.7
import numpy as np
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Lambda
from keras.layers import Input, average
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import ZeroPadding2D, Cropping2D
from keras import backend as K
from layer_common import mvn, crop
from keras.utils.vis_utils import plot_model

def dice_coef(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)


def jaccard_coef(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)

def dice_coef_each(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2)
    y_true_endo = y_true[:, :, :].astype('float32')
    y_pred_endo = y_pred[:, :, :]
    y_pred_endo = np.where(y_pred_endo > 0.5, 1.0, 0.0).astype('float32')
    intersection = np.sum(y_true_endo * y_pred_endo, axis=axes)
    summation = np.sum(y_true_endo * y_true_endo, axis=axes) + np.sum(y_pred_endo * y_pred_endo, axis=axes)

    return (2.0 * intersection + smooth) / (summation + smooth)

def fcn_model_inv(input_shape, num_classes, num_filter=64, weights=None):
    ''' "Skip" FCN architecture similar to Long et al., 2015
    https://arxiv.org/abs/1411.4038
    '''
    if num_classes == 2:
        num_classes = 1
        loss = dice_coef_loss
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
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
    mvn0 = Lambda(mvn, name='mvn0')(data)

    conv1 = Conv2D(filters=2**3*num_filter, name='conv1', **kwargs)(mvn0)
    mvn1 = Lambda(mvn, name='mvn1')(conv1)

    conv2 = Conv2D(filters=2**3*num_filter, name='conv2', **kwargs)(mvn1)
    mvn2 = Lambda(mvn, name='mvn2')(conv2)

    conv3 = Conv2D(filters=2**3*num_filter, name='conv3', **kwargs)(mvn2)
    mvn3 = Lambda(mvn, name='mvn3')(conv3)
    drop3 = Dropout(rate=0.5, name='drop1')(mvn3)
    pool1 = MaxPooling2D(pool_size=3, strides=2,
                    padding='same', name='pool1')(drop3)

    
    conv4 = Conv2D(filters=2**2*num_filter, name='conv4', **kwargs)(pool1)
    mvn4 = Lambda(mvn, name='mvn4')(conv4)

    conv5 = Conv2D(filters=2**2*num_filter, name='conv5', **kwargs)(mvn4)
    mvn5 = Lambda(mvn, name='mvn5')(conv5)

    conv6 = Conv2D(filters=2**2*num_filter, name='conv6', **kwargs)(mvn5)
    mvn6 = Lambda(mvn, name='mvn6')(conv6)

    conv7 = Conv2D(filters=2**2*num_filter, name='conv7', **kwargs)(mvn6)
    mvn7 = Lambda(mvn, name='mvn7')(conv7)
    drop7 = Dropout(rate=0.5, name='drop2')(mvn7)
    pool2 = MaxPooling2D(pool_size=3, strides=2,
                    padding='same', name='pool2')(drop7)


    conv8 = Conv2D(filters=2**1*num_filter, name='conv8', **kwargs)(pool2)
    mvn8 = Lambda(mvn, name='mvn8')(conv8)

    conv9 = Conv2D(filters=2**1*num_filter, name='conv9', **kwargs)(mvn8)
    mvn9 = Lambda(mvn, name='mvn9')(conv9)

    conv10 = Conv2D(filters=2**1*num_filter, name='conv10', **kwargs)(mvn9)
    mvn10 = Lambda(mvn, name='mvn10')(conv10)

    conv11 = Conv2D(filters=2**1*num_filter, name='conv11', **kwargs)(mvn10)
    mvn11 = Lambda(mvn, name='mvn11')(conv11)
    pool3 = MaxPooling2D(pool_size=3, strides=2,
                    padding='same', name='pool3')(mvn11)

    conv12 = Conv2D(filters=2**0*num_filter, name='conv12', **kwargs)(pool3)
    mvn12 = Lambda(mvn, name='mvn12')(conv12)

    conv13 = Conv2D(filters=2**0*num_filter, name='conv13', **kwargs)(mvn12)
    mvn13 = Lambda(mvn, name='mvn13')(conv13)

    conv14 = Conv2D(filters=2**0*num_filter, name='conv14', **kwargs)(mvn13)
    mvn14 = Lambda(mvn, name='mvn14')(conv14)

    conv15 = Conv2D(filters=2**0*num_filter, name='conv15', **kwargs)(mvn14)
    mvn15 = Lambda(mvn, name='mvn15')(conv15)

    score_conv15 = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=None, padding='same',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv15')(mvn15)
    upsample1 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                        strides=2, activation=None, padding='same',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample1')(score_conv15)
    score_conv11 = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=None, padding='same',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv11')(mvn11)
    crop1 = Lambda(crop, name='crop1')([upsample1, score_conv11])
    fuse_scores1 = average([crop1, upsample1], name='fuse_scores1')
    
    upsample2 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                        strides=2, activation=None, padding='same',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample2')(fuse_scores1)
    score_conv7 = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=None, padding='same',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv7')(drop7)
    crop2 = Lambda(crop, name='crop2')([upsample2, score_conv7])
    fuse_scores2 = average([crop2, upsample2], name='fuse_scores2')
    
    upsample3 = Conv2DTranspose(filters=num_classes, kernel_size=3,
                        strides=2, activation=None, padding='same',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample3')(fuse_scores2)
    score_conv3 = Conv2D(filters=num_classes, kernel_size=1,
                         strides=1, activation=None, padding='same',
                         kernel_initializer='glorot_uniform', use_bias=True,
                         name='score_conv3')(drop3)
    fuse_scores3 = average([score_conv3, upsample3], name='fuse_scores3')
    predictions = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=activation, padding='same',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='predictions')(fuse_scores3)
    
    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights, by_name=True)
    sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=loss,
                  metrics=['accuracy', dice_coef, jaccard_coef])

    return model


if __name__ == '__main__':
    model = fcn_model_inv((128, 128, 1), 2, num_filter=64, weights=None)
    plot_model(model, show_shapes=True, to_file='fcn_model_inv.png')
    model.summary()


