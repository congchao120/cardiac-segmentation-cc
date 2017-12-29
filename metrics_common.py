

from keras import backend as K
import tensorflow as tf
import numpy as np


def dice_coef(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient of endo/epi/rv per batch.'''

    return (
           dice_coef_endo(y_true, y_pred, smooth) + dice_coef_myo(y_true, y_pred, smooth) + dice_coef_rv(y_true, y_pred,
                                                                                                         smooth)) / 3.0


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)


def dice_coef_endo(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2)
    y_true_endo = y_true[:, :, :, 3]
    y_pred_endo = y_pred[:, :, :, 3]
    intersection = K.sum(y_true_endo * y_pred_endo, axis=axes)
    summation = K.sum(y_true_endo * y_true_endo, axis=axes) + K.sum(y_pred_endo * y_pred_endo, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_endo_each(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2)
    y_true_endo = y_true[:, :, :, 3].astype('float32')
    y_pred_endo = y_pred[:, :, :, 3]
    y_pred_endo = np.where(y_pred_endo > 0.5, 1.0, 0.0).astype('float32')
    intersection = np.sum(y_true_endo * y_pred_endo, axis=axes)
    summation = np.sum(y_true_endo * y_true_endo, axis=axes) + np.sum(y_pred_endo * y_pred_endo, axis=axes)

    return (2.0 * intersection + smooth) / (summation + smooth)

def dice_coef_loss_endo(y_true, y_pred):
    return 1.0 - dice_coef_endo(y_true, y_pred, smooth=0.0)


def dice_coef_myo(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for myocardium class per batch.'''
    axes = (1, 2)
    y_true_myo = y_true[:, :, :, 2]
    y_pred_myo = y_pred[:, :, :, 2]
    summation_true = K.sum(y_true_myo, axis=axes)
    intersection = K.sum(y_true_myo * y_pred_myo, axis=axes)
    summation = K.sum(y_true_myo * y_true_myo, axis=axes) + K.sum(y_pred_myo * y_pred_myo, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_myo_each(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2)
    y_true_myo = y_true[:, :, :, 2].astype('float32')
    y_pred_myo = y_pred[:, :, :, 2]
    y_pred_myo = np.where(y_pred_myo > 0.5, 1.0, 0.0).astype('float32')
    intersection = np.sum(y_true_myo * y_pred_myo, axis=axes)
    summation = np.sum(y_true_myo * y_true_myo, axis=axes) + np.sum(y_pred_myo * y_pred_myo, axis=axes)

    return (2.0 * intersection + smooth) / (summation + smooth)

def dice_coef_epi(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for myocardium class per batch.'''
    axes = (1, 2)
    y_true_myo = y_true[:, :, :, 2]
    y_pred_myo = y_pred[:, :, :, 2]
    y_true_endo = y_true[:, :, :, 3]
    y_pred_endo = y_pred[:, :, :, 3]

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

def summation_myo(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for myocardium class per batch.'''
    axes = (1, 2)
    y_true_myo = y_true[:, :, :, 2]
    summation_true = K.sum(y_true_myo, axis=axes)
    return summation_true


def dice_coef_loss_myo(y_true, y_pred):
    return 1.0 - K.minimum(dice_coef_myo(y_true, y_pred, smooth=1.0), dice_coef_endo(y_true, y_pred, smooth=1.0))

def dice_coef_rv(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for right ventricle per batch.'''
    axes = (1, 2)
    y_true_rv = y_true[:, :, :, 1]
    y_pred_rv = y_pred[:, :, :, 1]
    intersection = K.sum(y_true_rv * y_pred_rv, axis=axes)
    summation = K.sum(y_true_rv, axis=axes) + K.sum(y_pred_rv, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_loss_rv(y_true, y_pred):
    return 1.0 - dice_coef_rv(y_true, y_pred, smooth=10.0)


def jaccard_coef(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1, 2, 3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)



def dice_coef_each(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2)
    y_true_endo = y_true.astype('float32')
    y_pred_endo = y_pred
    y_pred_endo = np.where(y_pred_endo > 0.5, 1.0, 0.0).astype('float32')
    intersection = np.sum(y_true_endo * y_pred_endo, axis=axes)
    summation = np.sum(y_true_endo * y_true_endo, axis=axes) + np.sum(y_pred_endo * y_pred_endo, axis=axes)

    return (2.0 * intersection + smooth) / (summation + smooth)