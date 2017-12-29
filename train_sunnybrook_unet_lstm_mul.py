#!/usr/bin/env python2.7

import dicom, cv2, re
import os, fnmatch, sys
from keras.callbacks import *
from keras import backend as K
from itertools import zip_longest

from helpers import center_crop_3d, center_crop, lr_poly_decay, get_SAX_SERIES

import pylab
import matplotlib.pyplot as plt
from CardiacImageDataGenerator import CardiacImageDataGenerator
from unet_lstm_multi_model import unet_lstm_multi_model
seed = 1234
np.random.seed(seed)

SAX_SERIES = get_SAX_SERIES()
SUNNYBROOK_ROOT_PATH = 'D:\cardiac_data\Sunnybrook'

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
    h, w = img.shape
    classify = np.zeros((h, w, num_classes), dtype="uint8")

    coords = np.loadtxt(contour.ctr_endo_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)
    classify[...,2] = mask
    #classify[..., 2] = np.where(mask != 1, 1, 0)


    if os.path.exists(contour.ctr_epi_path):
        mask = np.zeros_like(img, dtype="uint8")
        coords = np.loadtxt(contour.ctr_epi_path, delimiter=' ').astype('int')
        cv2.fillPoly(mask, [coords], 1)
        classify[..., 1] = mask

    #classify[..., 1] = np.where(mask_union != 1 , 1, 0)
    classify[..., 0] = np.where(classify[..., 1] != 1 , 1, 0)
    classify[..., 1] = classify[..., 1] - classify[..., 2]

    img_arr = find_neighbor_images(contour, data_path, num_phases, num_phases_in_cycle, phase_dilation)

    if img_arr.ndim < 4:
        img_arr = img_arr[..., np.newaxis]
    if classify.ndim < 4:
        classify = classify[np.newaxis, ...]

    return img_arr, classify


def map_all_contours(contour_path):
    endo = []
    epi = []
    p1 = []
    p2 = []
    p3 = []
    for dirpath, dirnames, files in os.walk(contour_path):
        for epi_f in fnmatch.filter(files, 'IM-0001-*-ocontour-manual.txt'):
            epi.append(os.path.join(dirpath, epi_f))
            match = re.search(r'IM-0001-(\d{4})-ocontour-manual.txt', epi_f)  # it always has endo
            imgno = match.group(1)
            endo_f = 'IM-0001-' + imgno + '-icontour-manual.txt'
            p1_f = 'IM-0001-' + imgno + '-p1-manual.txt'
            p2_f = 'IM-0001-' + imgno + '-p2-manual.txt'
            p3_f = 'IM-0001-' + imgno + '-p3-manual.txt'
            endo.append(os.path.join(dirpath, endo_f))
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

def export_all_contours(contours, data_path, overlay_path, crop_size=100, num_classes=4, num_phases=5 ):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    images = np.zeros((len(contours), num_phases, crop_size, crop_size, 1))
    masks = np.zeros((len(contours), 1, crop_size, crop_size, num_classes))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path, num_classes, 5, 20, 2)
        #draw_contour(contour, data_path, overlay_path)

        img = center_crop_3d(img, crop_size=crop_size)
        mask = center_crop_3d(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask

    return images, masks

# ###############learning rate scheduler####################
def lr_scheduler(curr_epoch, curr_iter):
    total_iter = curr_epoch*steps_per_epoch + curr_iter
    lrate = lr_poly_decay(model, base_lr, total_iter, max_iter, power=0.5)

    print(' - lr: %f' % lrate)
    return lrate

if __name__== '__main__':

    contour_type = 'a'
    weight_path = None
    shuffle = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    crop_size = 128
    num_phases = 5
    save_path = 'model_logs'

    print('Mapping ground truth contours to images in train...')
    train_ctrs = list(map_all_contours(TRAIN_CONTOUR_PATH))
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(train_ctrs)

    print('Done mapping training set')
    num_classes = 3
    split = int(0.1*len(train_ctrs))
    dev_ctrs = train_ctrs[0:split]
    train_ctrs = train_ctrs[split:]
    
    print('\nBuilding Train dataset ...')
    img_train, mask_train = export_all_contours(train_ctrs,
                                                TRAIN_IMG_PATH,
                                                TRAIN_OVERLAY_PATH,
                                                crop_size=crop_size,
                                                num_classes=num_classes)
    print('\nBuilding Dev dataset ...')
    img_dev, mask_dev = export_all_contours(dev_ctrs,
                                            TRAIN_IMG_PATH,
                                            TRAIN_OVERLAY_PATH,
                                            crop_size=crop_size,
                                                num_classes=num_classes)
    
    input_shape = (num_phases, crop_size, crop_size, 1)

    #model = fcn_model(input_shape, num_classes, weights=weight_path)
    model = unet_lstm_multi_model(input_shape, num_classes, transfer=True, contour_type=contour_type, weights=weight_path)
    #plot_model(model, show_shapes=True, to_file='model.png')
    #model.summary()
    kwargs = dict(
        rotation_range=90,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        data_format="channels_last",
    )
    image_datagen = CardiacImageDataGenerator(**kwargs)
    mask_datagen = CardiacImageDataGenerator(**kwargs)
    aug_img_path = os.path.join(TRAIN_AUG_PATH, "Image")
    aug_mask_path = os.path.join(TRAIN_AUG_PATH, "Mask")
    img_train = image_datagen.fit_3d(img_train, augment=True, seed=seed, rounds=1, toDir=None)
    mask_train = mask_datagen.fit_3d(mask_train, augment=True, seed=seed, rounds=1, toDir=None)

    epochs = 50
    mini_batch_size = 1

    image_generator = image_datagen.flow3d(img_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    mask_generator = mask_datagen.flow3d(mask_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    train_generator = zip_longest(image_generator, mask_generator)

    dev_generator = (img_dev, mask_dev)
    
    max_iter = int(np.ceil(len(img_train) / mini_batch_size)) * epochs
    steps_per_epoch = int(np.ceil(len(img_train) / mini_batch_size))
    curr_iter = 0

    base_lr = K.eval(model.optimizer.lr)
    lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)

    callbacks = []
    # ####################### tfboard ###########################
    if K.backend() == 'tensorflow':
        tensorboard = TensorBoard(log_dir=os.path.join(save_path, 'logs_unet_multi'), histogram_freq=1, write_graph=False,
                                  write_grads=False, write_images=False)
        callbacks.append(tensorboard)
    # ################### checkpoint saver#######################
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'temp_weights.hdf5'),
                                 save_weights_only=True,
                                 save_best_only=False)  # .{epoch:d}
    callbacks.append(checkpoint)


    model.fit_generator(generator=train_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=dev_generator,
                        validation_steps=img_dev.__len__(),
                        epochs=epochs,
                        callbacks=callbacks,
                        workers=1,
                        class_weight=None
                        )
    save_file = '_'.join(['sunnybrook', contour_type, 'unet', 'multi']) + '.h5'
    save_file = os.path.join(save_path, save_file)
    model.save_weights(save_file)

    # for e in range(epochs):
    #     print('\nMain Epoch {:d}\n'.format(e + 1))
    #     print('\nLearning rate: {:6f}\n'.format(lrate))
    #     train_result = []
    #     for iteration in range(int(len(img_train) * augment_scale / mini_batch_size)):
    #         img, mask = next(train_generator)
    #         res = model.train_on_batch(img, mask)
    #         curr_iter += 1
    #         lrate = lr_poly_decay(model, base_lr, curr_iter,
    #                               max_iter, power=0.5)
    #         train_result.append(res)
    #     train_result = np.asarray(train_result)
    #     train_result = np.mean(train_result, axis=0).round(decimals=10)
    #     print('Train result {:s}:\n{:s}'.format(str(model.metrics_names), str(train_result)))
    #     print('\nEvaluating dev set ...')
    #     result = model.evaluate(img_dev, mask_dev, batch_size=32)
    #
    #     result = np.round(result, decimals=10)
    #     print('\nDev set result {:s}:\n{:s}'.format(str(model.metrics_names), str(result)))
    #     save_file = '_'.join(['sunnybrook', contour_type,
    #                           'epoch', str(e + 1)]) + '.h5'
    #     if not os.path.exists('model_logs'):
    #         os.makedirs('model_logs')
    #     save_path = os.path.join(save_path, save_file)
    #     print('\nSaving model weights to {:s}'.format(save_path))
    #     model.save_weights(save_path)
