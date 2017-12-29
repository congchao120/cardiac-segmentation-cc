#!/usr/bin/env python2.7
import dicom, cv2, re
import os, fnmatch, sys
import numpy as np
import cv2
from keras import backend as K
import pylab
import matplotlib.pyplot as plt

def get_SAX_SERIES():
    SAX_SERIES = {}
    with open('SAX_series.txt', 'r') as f:
        for line in f:
            if not line.startswith('#'):
                key, val = line.split(':')
                SAX_SERIES[key.strip()] = val.strip()
                #SAX_SERIES[key.strip()] = '0001' # modified by C.Cong 20170912

    return SAX_SERIES


def mvn(ndarray):
    '''Input ndarray is of rank 3 (height, width, depth).

    MVN performs per channel mean-variance normalization.
    '''
    epsilon = 1e-6
    mean = ndarray.mean(axis=(0,1), keepdims=True)
    std = ndarray.std(axis=(0,1), keepdims=True)

    return (ndarray - mean) / (std + epsilon)


def reshape(ndarray, to_shape):
    '''Reshapes a center cropped (or padded) array back to its original shape.'''
    h_in, w_in, d_in = ndarray.shape
    h_out, w_out, d_out = to_shape
    if h_in > h_out: # center crop along h dimension
        h_offset = (h_in - h_out) / 2
        ndarray = ndarray[h_offset:(h_offset+h_out), :, :]
    else: # zero pad along h dimension
        pad_h = (h_out - h_in)
        rem = pad_h % 2
        pad_dim_h = (int(pad_h/2), int(pad_h/2) + rem)
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, (0,0), (0,0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
    if w_in > w_out: # center crop along w dimension
        w_offset = (w_in - w_out) / 2
        ndarray = ndarray[:, w_offset:(w_offset+w_out), :]
    else: # zero pad along w dimension
        pad_w = (w_out - w_in)
        rem = pad_w % 2
        pad_dim_w = (int(pad_w/2), int(pad_w/2) + rem)
        npad = ((0,0), pad_dim_w, (0,0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
    
    return ndarray # reshaped


def center_crop(ndarray, crop_size):
    '''Input ndarray is of rank 3 (height, width, depth).

    Argument crop_size is an integer for square cropping only.

    Performs padding and center cropping to a specified size.
    '''
    h, w, d = ndarray.shape
    if crop_size == 0:
        raise ValueError('argument crop_size must be non-zero integer')

    if any([dim < crop_size for dim in (h, w)]):
        # zero pad along each (h, w) dimension before center cropping
        pad_h = (crop_size - h) if (h < crop_size) else 0
        pad_w = (crop_size - w) if (w < crop_size) else 0
        rem_h = pad_h % 2
        rem_w = pad_w % 2
        pad_dim_h = (int(pad_h/2), int(pad_h/2 + rem_h))
        pad_dim_w = (int(pad_w/2), int(pad_w/2 + rem_w))
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, pad_dim_w, (0,0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
        h, w, d = ndarray.shape
    # center crop
    h_offset = int((h - crop_size) / 2)
    w_offset = int((w - crop_size) / 2)
    cropped = ndarray[h_offset:(h_offset+crop_size),
                      w_offset:(w_offset+crop_size), :]

    return cropped


def center_crop_3d(ndarray, crop_size):
    '''Input ndarray is of rank 4 (phase, height, width, depth).

    Argument crop_size is an integer for square cropping only.

    Performs padding and center cropping to a specified size.
    '''
    p, h, w, d = ndarray.shape
    if crop_size == 0:
        raise ValueError('argument crop_size must be non-zero integer')

    if any([dim < crop_size for dim in (h, w)]):
        # zero pad along each (h, w) dimension before center cropping
        pad_h = (crop_size - h) if (h < crop_size) else 0
        pad_w = (crop_size - w) if (w < crop_size) else 0
        rem_h = pad_h % 2
        rem_w = pad_w % 2
        pad_dim_h = (int(pad_h / 2), int(pad_h / 2 + rem_h))
        pad_dim_w = (int(pad_w / 2), int(pad_w / 2 + rem_w))
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (p, pad_dim_h, pad_dim_w, (0, 0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
        p, h, w, d = ndarray.shape
    # center crop
    h_offset = int((h - crop_size) / 2)
    w_offset = int((w - crop_size) / 2)
    cropped = ndarray[:, h_offset:(h_offset + crop_size),
              w_offset:(w_offset + crop_size), :]

    return cropped

def lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5):
    lrate = base_lr * (1.0 - (curr_iter / float(max_iter)))**power
    K.set_value(model.optimizer.lr, lrate)

    return K.eval(model.optimizer.lr)


def dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred, axis=None)
    summation = np.sum(y_true, axis=None) + np.sum(y_pred, axis=None)
    
    return 2.0 * intersection / summation


def jaccard_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred, axis=None)
    union = np.sum(y_true, axis=None) + np.sum(y_pred, axis=None) - intersection

    return float(intersection) / float(union)



def draw_result(contour, data_path, out_path, contour_type, coords):
    # filename = 'IM-%s-%04d.dcm' % (SAX_SERIES[contour.case], contour.img_no)

    filename = 'IM-0001-%04d.dcm' % (contour.img_no)
    outname = 'IM-0001-%s-%04d.png' % (contour_type, contour.img_no)
    full_path = os.path.join(data_path, contour.case, 'DICOM', filename)  # modified by C.Cong
    out_full_path = os.path.join(out_path, contour.case)  # modified by C.Cong
    out_full_name = os.path.join(out_full_path, outname)
    if not os.path.exists(out_full_path):
        os.makedirs(out_full_path)
    f = dicom.read_file(full_path)
    img = f.pixel_array
    img_size = img.shape
    plt.cla()
    pylab.imshow(img, cmap=pylab.cm.bone)

    if isinstance(coords, np.ndarray):
        coords = np.squeeze(coords)
        if coords.ndim == 1:
            x, y = coords
        else:
            x, y = zip(*coords)

        if contour_type == 'i':
            plt.plot(x, y, 'r.')
        elif contour_type == 'm' or contour_type == 'o':
            plt.plot(x, y, 'b.')
    elif len(coords) > 1:
        for cd in coords:
            cd = np.squeeze(cd)
            if cd.ndim == 1:
                x, y = cd
            else:
                x, y = zip(*cd)

            if contour_type == 'i':
                plt.plot(x, y, 'r.')
            elif contour_type == 'm' or contour_type == 'o':
                plt.plot(x, y, 'b.')
    elif len(coords) == 1:
        coords = np.squeeze(coords)
        if coords.ndim == 1:
            x, y = coords
        else:
            x, y = zip(*coords)

        if contour_type == 'i':
            plt.plot(x, y, 'r.')
        elif contour_type == 'm' or contour_type == 'o':
            plt.plot(x, y, 'b.')


    plt.xlim(50, img_size[0]-50)
    plt.ylim(50, img_size[1]-50)
    pylab.savefig(out_full_name,bbox_inches='tight',dpi=200)

    #pylab.show()
    return


def draw_image_overlay(image, out_file, out_path, contour_type, coords):
    out_full_name = os.path.join(out_path, out_file)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    image = np.squeeze(image)
    plt.cla()
    pylab.imshow(image, cmap=pylab.cm.bone)

    if isinstance(coords, np.ndarray):
        coords = np.squeeze(coords)
        if coords.ndim == 1:
            x, y = coords
        else:
            x, y = zip(*coords)

        if contour_type == 'i':
            plt.plot(x, y, 'r.')
        elif contour_type == 'm' or contour_type == 'o':
            plt.plot(x, y, 'b.')
    elif len(coords) > 1:
        for cd in coords:
            cd = np.squeeze(cd)
            if cd.ndim == 1:
                x, y = cd
            else:
                x, y = zip(*cd)

            if contour_type == 'i':
                plt.plot(x, y, 'r.')
            elif contour_type == 'm' or contour_type == 'o':
                plt.plot(x, y, 'b.')
    elif len(coords) == 1:
        coords = np.squeeze(coords)
        if coords.ndim == 1:
            x, y = coords
        else:
            x, y = zip(*coords)

        if contour_type == 'i':
            plt.plot(x, y, 'r.')
        elif contour_type == 'm' or contour_type == 'o':
            plt.plot(x, y, 'b.')

    pylab.savefig(out_full_name,bbox_inches='tight',dpi=200)

    #pylab.show()
    return

