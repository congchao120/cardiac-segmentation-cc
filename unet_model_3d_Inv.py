import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Lambda, Dropout
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from functools import partial
from layer_common import mvn3d, mvn
from keras.models import load_model
try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

def dice_coef_endo(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2, 3)
    y_true_endo = y_true[..., 2]
    y_pred_endo = y_pred[..., 2]
    intersection = K.sum(y_true_endo * y_pred_endo, axis=axes)
    summation = K.sum(y_true_endo, axis=axes) + K.sum(y_pred_endo, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)

def dice_coef_myo(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for myocardium class per batch.'''
    axes = (1, 2, 3)
    y_true_myo = y_true[..., 1]
    y_pred_myo = y_pred[..., 1]
    intersection = K.sum(y_true_myo * y_pred_myo, axis=axes)
    summation = K.sum(y_true_myo, axis=axes) + K.sum(y_pred_myo, axis=axes)

    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)

def dice_coef_endo_each(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2)
    y_true_endo = y_true[..., 2].astype('float32')
    y_pred_endo = y_pred[..., 2]
    y_pred_endo = np.where(y_pred_endo > 0.5, 1.0, 0.0).astype('float32')
    intersection = np.sum(y_true_endo * y_pred_endo, axis=axes)
    summation = np.sum(y_true_endo, axis=axes) + np.sum(y_pred_endo, axis=axes)

    return (2.0 * intersection + smooth) / (summation + smooth)

def dice_coef_myo_each(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient for endocardium class per batch.'''
    axes = (1, 2)
    y_true_myo = y_true[..., 1].astype('float32')
    y_pred_myo = y_pred[..., 1]
    y_pred_myo = np.where(y_pred_myo > 0.5, 1.0, 0.0).astype('float32')
    intersection = np.sum(y_true_myo * y_pred_myo, axis=axes)
    summation = np.sum(y_true_myo, axis=axes) + np.sum(y_pred_myo, axis=axes)

    return (2.0 * intersection + smooth) / (summation + smooth)

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_cardiac_loss(y_true, y_pred):
    return 2 - (dice_coef_endo(y_true, y_pred) + dice_coef_myo(y_true, y_pred))

def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coef(y_true[..., label_index], y_pred[..., label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

def unet_model_3d_Inv(input_shape, pool_size=(2, 2, 2), n_labels=1, kernel=(3, 3, 3), initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coef,
                  batch_normalization=False, weights=None):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**(depth-layer_depth)),
                                          batch_normalization=batch_normalization, kernel=kernel)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**(depth-layer_depth)),
                                          batch_normalization=batch_normalization, kernel=kernel)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution, depth=layer_depth,
                                            n_filters=current_layer._keras_shape[1],
                                            image_shape=input_shape[-3:])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=4)
        current_layer = create_convolution_block(n_filters=levels[depth-layer_depth-1][1]._keras_shape[4],
                                                 input_layer=concat, batch_normalization=batch_normalization, kernel=kernel)
        current_layer = create_convolution_block(n_filters=levels[depth-layer_depth-1][1]._keras_shape[4],
                                                 input_layer=current_layer, batch_normalization=batch_normalization, kernel=kernel)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation('softmax')(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        if metrics:
            metrics = metrics + [dice_coef_endo, dice_coef_myo]
        else:
            metrics = [dice_coef_endo, dice_coef_myo]

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_cardiac_loss, metrics=metrics)
    if(weights != None):
        model.load_weights(weights)
    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same'):
    """
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    mvn0 = Lambda(mvn)(input_layer)
    layer = Conv3D(n_filters, kernel, padding=padding)(mvn0)
    if batch_normalization:
        layer = BatchNormalization(axis=4)(layer)
    layer = Dropout(rate=0.2)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(depth, n_filters, pool_size, image_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        try:
            from keras_contrib.layers import Deconvolution3D
        except ImportError:
            raise ImportError("Install keras_contrib in order to use deconvolution. Otherwise set deconvolution=False."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")

        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               output_shape=compute_level_output_shape(n_filters=n_filters, depth=depth,
                                                                       pool_size=pool_size, image_shape=image_shape),
                               strides=strides, input_shape=compute_level_output_shape(n_filters=n_filters,
                                                                                       depth=depth,
                                                                                       pool_size=pool_size,
                                                                                       image_shape=image_shape))
    else:
        return UpSampling3D(size=pool_size)


def resume_training(model_file):
    print("Resume training and load model")
    custom_objects = {'dice_coef_cardiac_loss': dice_coef_cardiac_loss, 'dice_coef': dice_coef,
                      'dice_coef_endo': dice_coef_endo,
                      'dice_coef_myo':dice_coef_endo}
    return load_model(model_file, custom_objects=custom_objects)

if __name__ == '__main__':
    model = unet_model_3d_Inv((128, 128, 5, 1), pool_size=(2, 2, 1), kernel=(7, 7, 5), n_labels=3, initial_learning_rate=0.00001,
                              deconvolution=False, depth=4, n_base_filters=8, include_label_wise_dice_coefficients=True, batch_normalization=True)
    plot_model(model, show_shapes=True, to_file='unet_model_3d_Inv.png')
    model.summary()