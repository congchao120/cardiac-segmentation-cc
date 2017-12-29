#!/usr/bin/env python2.7

import dicom, cv2, re
import os, fnmatch, sys
from keras.callbacks import *
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from itertools import zip_longest
from scipy.misc import imsave
from helpers import center_crop_3d, center_crop, lr_poly_decay, get_SAX_SERIES

import pylab
import matplotlib.pyplot as plt
from CardiacImageDataGenerator import CardiacImageDataGenerator, CardiacTimeSeriesDataGenerator
from unet_model_time import unet_res_model_time
from unet_res_model_Inv import unet_res_model_Inv
from DataIOProc import DataIOProc
seed = 1234
np.random.seed(seed)

SAX_SERIES = get_SAX_SERIES()
ACDC_ROOT_PATH = 'D:\cardiac_data\ACDC'

TEMP_CONTOUR_PATH = os.path.join(ACDC_ROOT_PATH,
                   'ACDC Cardiac MR Database Temp',
                   'Temp')


class Contour(object):
    def __init__(self, ctr_endo_path, ctr_epi_path, ctr_p1_path, ctr_p2_path, ctr_p3_path):
        self.ctr_endo_path = ctr_endo_path
        self.ctr_epi_path = ctr_epi_path
        self.ctr_p1_path = ctr_p1_path
        self.ctr_p2_path = ctr_p2_path
        self.ctr_p3_path = ctr_p3_path
        match = re.search(r'\\([^\\]*)\\contours-manual\\IRCCI-expert\\IM-0001-(\d{4})-.*', ctr_endo_path) #it always has endo
        self.case = match.group(1)
        self.img_no = int(match.group(2))
    
    def __str__(self):
        return '<Contour for case %s, image %d>' % (self.case, self.img_no)
    
    __repr__ = __str__

def find_neighbor_images(contour, data_path, num_phases, num_phases_in_cycle, phase_dilation):
    center_index = contour.img_no
    center_file = 'IM-0001-%04d.dcm' % (contour.img_no)
    center_file_path = os.path.join(data_path, contour.case, 'DICOM', center_file) #modified by C.Cong
    center = dicom.read_file(center_file_path)
    center_slice_pos = center[0x20, 0x1041]
    center_img = center.pixel_array.astype('int')
    h, w = center_img.shape
    img_arr = np.zeros((num_phases, h, w), dtype="int")
    for i in range (num_phases):
        idx = int(center_index + (i - int(num_phases/2))*phase_dilation)
        filename = 'IM-0001-%04d.dcm' % (idx)
        full_path = os.path.join(data_path, contour.case, 'DICOM', filename)
        #If
        if os.path.isfile(full_path) == False:
            if idx < center_index:
                idx = idx + num_phases_in_cycle
                filename = 'IM-0001-%04d.dcm' % (idx)
                full_path = os.path.join(data_path, contour.case, 'DICOM', filename)
            else:
                idx = idx - num_phases_in_cycle
                filename = 'IM-0001-%04d.dcm' % (idx)
                full_path = os.path.join(data_path, contour.case, 'DICOM', filename)

        f = dicom.read_file(full_path)
        f_slice_pos = f[0x20, 0x1041]

        if(f_slice_pos.value != center_slice_pos.value):
            idx = idx + num_phases_in_cycle
            filename = 'IM-0001-%04d.dcm' % (idx)
            full_path = os.path.join(data_path, contour.case, 'DICOM', filename)
            if os.path.isfile(full_path) == True:
                f = dicom.read_file(full_path)
                f_slice_pos = f[0x20, 0x1041]

        if (f_slice_pos.value != center_slice_pos.value):
            idx = idx - num_phases_in_cycle - num_phases_in_cycle
            filename = 'IM-0001-%04d.dcm' % (idx)
            full_path = os.path.join(data_path, contour.case, 'DICOM', filename)
            if os.path.isfile(full_path) == True:
                f = dicom.read_file(full_path)
                f_slice_pos = f[0x20, 0x1041]

        if (f_slice_pos.value != center_slice_pos.value):
            raise AssertionError('Cannot find neighbor files for: {:s}'.format(center_file_path))

        img_arr[i] = f.pixel_array.astype('int')

    return img_arr

def read_contour(contour, data_path, num_classes, num_phases, num_phases_in_cycle, phase_dilation):
    #filename = 'IM-%s-%04d.dcm' % (SAX_SERIES[contour.case], contour.img_no)
    filename = 'IM-0001-%04d.dcm' % (contour.img_no)
    full_path = os.path.join(data_path, contour.case, 'DICOM', filename) #modified by C.Cong
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype="uint8")

    coords = np.loadtxt(contour.ctr_endo_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)
    classify = mask

    img_arr = find_neighbor_images(contour, data_path, num_phases, num_phases_in_cycle, phase_dilation)

    if img_arr.ndim < 4:
        img_arr = img_arr[..., np.newaxis]
    if classify.ndim < 4:
        classify = classify[np.newaxis, ..., np.newaxis]

    return img_arr, classify


def map_all_contours(contour_path):
    endo = []
    epi = []
    p1 = []
    p2 = []
    p3 = []
    for dirpath, dirnames, files in os.walk(contour_path):
        for endo_f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt'):
            endo.append(os.path.join(dirpath, endo_f))
            match = re.search(r'IM-0001-(\d{4})-icontour-manual.txt', endo_f)  # it always has endo
            imgno = match.group(1)
            epi_f = 'IM-0001-' + imgno + '-ocontour-manual.txt'
            p1_f = 'IM-0001-' + imgno + '-p1-manual.txt'
            p2_f = 'IM-0001-' + imgno + '-p2-manual.txt'
            p3_f = 'IM-0001-' + imgno + '-p3-manual.txt'
            epi.append(os.path.join(dirpath, epi_f))
            p1.append(os.path.join(dirpath, p1_f))
            p2.append(os.path.join(dirpath, p2_f))
            p3.append(os.path.join(dirpath, p3_f))

    print('Number of examples: {:d}'.format(len(endo)))
    contours = map(Contour, endo, epi, p1, p2, p3)

    return contours


def map_endo_contours(contour_path):
    endo = []
    epi = []
    p1 = []
    p2 = []
    p3 = []
    for dirpath, dirnames, files in os.walk(contour_path):
        for endo_f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt'):
            endo.append(os.path.join(dirpath, endo_f))
            match = re.search(r'IM-0001-(\d{4})-icontour-manual.txt', endo_f)  # it always has endo
            imgno = match.group(1)
            epi_f = 'IM-0001-' + imgno + '-ocontour-manual.txt'
            p1_f = 'IM-0001-' + imgno + '-p1-manual.txt'
            p2_f = 'IM-0001-' + imgno + '-p2-manual.txt'
            p3_f = 'IM-0001-' + imgno + '-p3-manual.txt'
            epi.append(os.path.join(dirpath, epi_f))
            p1.append(os.path.join(dirpath, p1_f))
            p2.append(os.path.join(dirpath, p2_f))
            p3.append(os.path.join(dirpath, p3_f))

    print('Number of examples: {:d}'.format(len(endo)))
    contours = map(Contour, endo, epi, p1, p2, p3)

    return contours

def export_all_contours(contours, data_path, overlay_path, crop_size=100, num_classes=4, num_phases=5, phase_dilation=1):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    if num_classes == 2:
        num_classes = 1
    images = np.zeros((len(contours), num_phases, crop_size, crop_size, 1))
    masks = np.zeros((len(contours), 1, crop_size, crop_size, num_classes))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path, num_classes, num_phases, 20, phase_dilation)
        #draw_contour(contour, data_path, overlay_path)

        img = center_crop_3d(img, crop_size=crop_size)
        mask = center_crop_3d(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask

    return images, masks

if __name__== '__main__':

    contour_type = 'a'
    weight_s = 'model_logs/sunnybrook_i_unetres_inv.h5'
    shuffle = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    crop_size = 128
    num_phases = 5
    save_path = 'model_logs'
    phase_dilation = 1
    data_proc = DataIOProc(TEMP_CONTOUR_PATH, 'p5_a4')
    num_classes = 2
    s = 6800
    p = 5
    h = 128
    w = 128
    d = 1
    s_val = 202
    p_val = 5
    h_val = 128
    w_val = 128
    d_val = 1
    print('\nPredict for 2nd training ...')
    # Load training dataset
    temp_mask_t = data_proc.load_data_4d('training_data.bin', s, p, h, w, d)
    mask_train = data_proc.load_data_4d('training_mask.bin', s, 1, h, w, d)

    # Load validation dataset
    print('\nTotal sample is {:d} for 2nd training.'.format(s))
    #print('\nPredict for 2nd evaluating ...')
    temp_mask_dev = data_proc.load_data_4d('eval_data.bin', s_val, p_val, h_val, w_val, d_val)
    mask_dev = data_proc.load_data_4d('eval_mask.bin', s_val, 1, h_val, w_val, d_val)
    dev_generator = (temp_mask_dev, mask_dev)

    input_shape = (num_phases, crop_size, crop_size, 1)
    epochs = 100
    model_t = unet_res_model_time(input_shape, num_classes, nb_filters=32, n_phases=num_phases, dilation=phase_dilation, transfer=True, weights=None)

    callbacks = []
    # ####################### tfboard ###########################
    if K.backend() == 'tensorflow':
        tensorboard = TensorBoard(log_dir=os.path.join(save_path, 'logs_acdc_unet_time'), histogram_freq=0, write_graph=False,
                                  write_grads=False, write_images=False)
        callbacks.append(tensorboard)
    # ################### checkpoint saver#######################
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'check_point_model.hdf5'),
                                 save_weights_only=False,
                                 save_best_only=False,
                                 period=10)  # .{epoch:d}
    callbacks.append(checkpoint)



    print('\nTotal sample is {:d} for 2nd evaluation.'.format(s_val))
    mini_batch_size = 1
    steps_per_epoch = int(np.ceil(s / mini_batch_size))
    model_t.fit(temp_mask_t,
                mask_train,
                epochs=epochs,
                batch_size=16,
                validation_data=dev_generator,
                #validation_steps=mask_dev.__len__(),
                callbacks=callbacks,
                class_weight=None
                )


    save_file = '_'.join(['acdc', contour_type, 'unetres_inv_time']) + '.h5'
    save_file = os.path.join(save_path, save_file)
    model_t.save_weights(save_file)
