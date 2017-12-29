from keras import backend as K
from keras.layers import ZeroPadding2D, Cropping2D

def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1, 2), keepdims=True)
    std = K.std(tensor, axis=(1, 2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)

    return mvn

def mvn3d(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1, 2, 3), keepdims=True)
    std = K.std(tensor, axis=(1, 2, 3), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)

    return mvn

def crop(tensors):
    '''
    List of 2 tensors, the second tensor having larger spatial dimensions.
    '''
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.get_variable_shape(t)
        h_dims.append(h)
        w_dims.append(w)
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (int(crop_h / 2), int(crop_h / 2 + rem_h))
    crop_w_dims = (int(crop_w / 2), int(crop_w / 2 + rem_w))
    cropped = Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1])

    return cropped