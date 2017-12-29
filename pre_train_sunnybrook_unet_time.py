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
SUNNYBROOK_ROOT_PATH = 'D:\cardiac_data\Sunnybrook'

TEMP_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'Sunnybrook Cardiac MR Database Temp',
                   'Temp')
TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart3',
                            'TrainingDataContours')
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook Cardiac MR Database DICOMPart3',
                              'TrainingDataDICOM')
TRAIN_OVERLAY_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook Cardiac MR Database OverlayPart3',
                              'TrainingOverlayImage')

TRAIN_AUG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook Cardiac MR Database Augmentation')

DEBUG_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart3',
                            'Debug')
DEBUG_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook Cardiac MR Database DICOMPart3',
                              'Debug')
DEBUG_OVERLAY_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook Cardiac MR Database OverlayPart3',
                              'Debug')
VAL_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'Sunnybrook Cardiac MR Database ContoursPart2',
                   'ValidationDataContours')
VAL_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'Sunnybrook Cardiac MR Database DICOMPart2',
                    'ValidationDataDICOM')
VAL_OVERLAY_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'Sunnybrook Cardiac MR Database OverlayPart2',
                    'ValidationDataOverlay')
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

# ###############learning rate scheduler####################
def lr_scheduler(curr_epoch, curr_iter):
    total_iter = curr_epoch*steps_per_epoch + curr_iter
    lrate = lr_poly_decay(model_s, base_lr, total_iter, max_iter, power=0.5)

    print(' - lr: %f' % lrate)
    return lrate

if __name__== '__main__':

    contour_type = 'a'
    weight_s = 'model_logs/sunnybrook_i_unetres_inv_drop_acdc.h5'
    shuffle = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    crop_size = 128
    num_phases = 5
    save_path = 'model_logs'
    phase_dilation = 4
    data_proc = DataIOProc(TEMP_CONTOUR_PATH, 'p5_a4')

    print('Mapping ground truth contours to images in train...')
    train_ctrs = list(map_all_contours(TRAIN_CONTOUR_PATH))
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(train_ctrs)

    print('Done mapping training set')
    num_classes = 2
    #No dev
    split = int(0.1*len(train_ctrs))
    dev_ctrs = train_ctrs[0:split]
    train_ctrs = train_ctrs[split:]
    
    print('\nBuilding Train dataset ...')
    img_train, mask_train = export_all_contours(train_ctrs,
                                                TRAIN_IMG_PATH,
                                                TRAIN_OVERLAY_PATH,
                                                crop_size=crop_size,
                                                num_classes=num_classes,
                                                num_phases=num_phases,
                                                phase_dilation=phase_dilation)
    print('\nBuilding Dev dataset ...')
    img_dev, mask_dev = export_all_contours(dev_ctrs,
                                            TRAIN_IMG_PATH,
                                            TRAIN_OVERLAY_PATH,
                                            crop_size=crop_size,
                                            num_classes=num_classes,
                                            num_phases=num_phases,
                                            phase_dilation=phase_dilation)


    input_shape = (num_phases, crop_size, crop_size, 1)
    input_shape_s = (crop_size, crop_size, 1)
    model_s = unet_res_model_Inv(input_shape_s, num_classes, nb_filters=8, transfer=True, contour_type=contour_type, weights=weight_s)

    kwargs = dict(
        rotation_range=90,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        data_format="channels_last",
    )
    image_datagen = CardiacTimeSeriesDataGenerator(**kwargs)
    mask_datagen = CardiacTimeSeriesDataGenerator(**kwargs)
    aug_img_path = os.path.join(TRAIN_AUG_PATH, "Image")
    aug_mask_path = os.path.join(TRAIN_AUG_PATH, "Mask")
    img_train = image_datagen.fit(img_train, augment=True, seed=seed, rounds=8, toDir=None)
    mask_train = mask_datagen.fit(mask_train, augment=True, seed=seed, rounds=8, toDir=None)
    epochs = 200
    mini_batch_size = 4
    s, p, h, w, d = img_train.shape
    s_val, p_val, h_val, w_val, d_val = img_dev.shape
    max_iter = int(np.ceil(len(img_train) / mini_batch_size)) * epochs
    steps_per_epoch = int(np.ceil(len(img_train) / mini_batch_size))
    curr_iter = 0

    base_lr = K.eval(model_s.optimizer.lr)
    lrate = lr_poly_decay(model_s, base_lr, curr_iter, max_iter, power=0.5)

    callbacks = []
    # ####################### tfboard ###########################
    if K.backend() == 'tensorflow':
        tensorboard = TensorBoard(log_dir=os.path.join(save_path, 'logs_unet_time'), histogram_freq=1, write_graph=False,
                                  write_grads=False, write_images=False)
        callbacks.append(tensorboard)
    # ################### checkpoint saver#######################
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'check_point_model.hdf5'),
                                 save_weights_only=False,
                                 save_best_only=False,
                                 period=2)  # .{epoch:d}
    callbacks.append(checkpoint)

    print('\nPredict for 2nd training ...')
    #img_train_s = img_train[:,4,...]
    #mask_train_s = mask_train[:,0,...]
    #result = model_s.evaluate(img_train_s, mask_train_s)
    #result = np.round(result, decimals=10)
    #print('\nDev set result {:s}:\n{:s}'.format(str(model_s.metrics_names), str(result)))
    if not os.path.exists(TEMP_CONTOUR_PATH):
        os.makedirs(TEMP_CONTOUR_PATH)

    # Create training dataset
    temp_image_t = np.reshape(img_train, (s*p, h, w, d))
    temp_mask_t = model_s.predict(temp_image_t, batch_size=32, verbose=1)
    temp_mask_t = np.reshape(temp_mask_t, (s, p, h, w, d))

    data_proc.save_image_4d(temp_mask_t, 'training')
    data_proc.save_image_4d(mask_train, 'training_mask')
    data_proc.save_data_4d(temp_mask_t.astype('float32'), 'training_data.bin')
    data_proc.save_data_4d(mask_train.astype('float32'), 'training_mask.bin')

    # train_mask_p = np.zeros((s, p, w, h, 1), dtype=K.floatx())
    # for idx_s in range(s):
    #     img_train_p = img_train[idx_s,...]
    #     train_mask_p[idx_s] = model_s.predict(img_train_p)
        #
        # for idx_p in range(p):
        #     mask = train_mask_p[idx_s, idx_p, ...]
        #     img = img_train[idx_s, idx_p, ...]
        #     img = np.squeeze(img*mask)
        #     img_name = '{:d}-{:d}'.format(idx_s, idx_p)
        #     imsave(os.path.join(TEMP_CONTOUR_PATH, img_name + ".png"), img)
    # Create validation dataset
    print('\nTotal sample is {:d} for 2nd training.'.format(s))
    print('\nPredict for 2nd evaluating ...')
    temp_image_dev = np.reshape(img_dev, (s_val*p_val, w_val, h_val, d_val))
    temp_mask_dev = model_s.predict(temp_image_dev, batch_size=16, verbose=1)
    temp_mask_dev = np.reshape(temp_mask_dev, (s_val, p_val, w_val, h_val, d_val))

    data_proc.save_image_4d(temp_mask_dev, 'evaluation')
    data_proc.save_image_4d(mask_dev, 'evaluation_mask')

    data_proc.save_data_4d(temp_mask_dev.astype('float32'), 'eval_data.bin')
    data_proc.save_data_4d(mask_dev.astype('float32'), 'eval_mask.bin')
    #print('\nTotal sample is {:d} for 2nd evaluation.'.format(s_val))
    # val_mask_p = np.zeros((s_val, p_val, w_val, h_val, 1), dtype=K.floatx())
    # for idx_s in range(s_val):
    #     img_val_p = img_dev[idx_s,...]
    #     val_mask_p[idx_s] = model_s.predict(img_val_p)


    # dev_generator = (temp_mask_dev, mask_dev)
    # print('\nTotal sample is {:d} for 2nd evaluation.'.format(s_val))
    # model_t = unet_res_model_time(input_shape, num_classes, nb_filters=64, n_phases=num_phases, dilation=phase_dilation, transfer=True, weights=None)
    # model_t.fit(temp_mask_t,
    #             mask_train,
    #             epochs=epochs,
    #             batch_size=1,
    #             validation_data=dev_generator,
    #             callbacks=callbacks,
    #             class_weight=None
    #             )
