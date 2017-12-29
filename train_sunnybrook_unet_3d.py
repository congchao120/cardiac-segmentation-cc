#!/usr/bin/env python2.7

import dicom, cv2, re
import os, fnmatch, sys
from keras.callbacks import *
from keras import backend as K
from itertools import zip_longest
from helpers import center_crop_3d, center_crop, lr_poly_decay, get_SAX_SERIES

import pylab
import matplotlib.pyplot as plt
from CardiacImageDataGenerator import CardiacImageDataGenerator, CardiacVolumeDataGenerator
from unet_model_3d import unet_model_3d, resume_training
from unet_model_3d_Inv import unet_model_3d_Inv, resume_training
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

def read_mask(contour, data_path, num_classes):
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

    return classify

def read_image(img_no, data_path, case):
    filename = 'IM-0001-%04d.dcm' % (img_no)
    full_path = os.path.join(data_path, case, 'DICOM', filename)
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype('int')

    if img.ndim < 3:
        img = img[..., np.newaxis]

    return img

def find_min_max_image(data_path, case):
    full_path = os.path.join(data_path, case, 'DICOM')
    min_no = 9999
    max_no = 0
    for dirpath, dirnames, files in os.walk(full_path):
        for file in files:
            match = re.search(r'IM-0001-(\d{4}).dcm', file)  # it always has endo
            if match != None:
                imgno = int(match.group(1))
                if min_no > imgno:
                    min_no = imgno
                if max_no < imgno:
                    max_no = imgno

    return min_no, max_no

def read_volume(center_ctr, volume_map, data_path, num_classes, num_slices, num_phases_in_cycle, crop_size, is_all_valid_slice):
    case = center_ctr.case
    center_no = center_ctr.img_no
    img_index = center_ctr.img_no % num_phases_in_cycle
    if img_index == 0:
        img_index = num_phases_in_cycle

    img_no_min, img_no_max = find_min_max_image(data_path, case)
    images = np.zeros((crop_size, crop_size, num_slices, 1))
    masks = np.zeros((crop_size, crop_size, num_slices, num_classes))
    masks_bg = np.ones((crop_size, crop_size, num_slices))
    masks[:,:,:,0] = masks_bg
    if is_all_valid_slice:
        for slice_idx in range(num_slices):
            img_no = center_no + (slice_idx - int(num_slices / 2)) * num_phases_in_cycle
            if img_no not in volume_map[case]:
                return [], []

    for slice_idx in range(num_slices):
        img_no = center_no + (slice_idx - int(num_slices/2))*num_phases_in_cycle
        if img_no < img_no_min:
            img_no = img_no_min
        if img_no > img_no_max:
            img_no = img_no_max
        img = read_image(img_no, data_path, case)
        img = center_crop(img, crop_size)
        images[:,:,slice_idx,:] = img
        if img_no in volume_map[case]:
            mask = read_mask(volume_map[case][img_no], data_path, num_classes)
            mask = center_crop(mask, crop_size)
            masks[:, :, slice_idx, :] = mask


    return images, masks


def map_all_contours(contour_path):
    endo = []
    epi = []
    p1 = []
    p2 = []
    p3 = []
    volume_map = {}
    contour_map = {}
    for dirpath, dirnames, files in os.walk(contour_path):
        contour_map = {}

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

            contour_map[int(imgno)] = Contour(os.path.join(dirpath, endo_f), os.path.join(dirpath, epi_f), os.path.join(dirpath, p1_f), os.path.join(dirpath, p2_f), os.path.join(dirpath, p3_f))

        match = re.search(r'\\([^\\]*)\\contours-manual\\IRCCI-expert', dirpath)
        if(match != None):
            case = match.group(1)
            volume_map[case] = contour_map

    print('Number of examples: {:d}'.format(len(endo)))
    contours = list(map(Contour, endo, epi, p1, p2, p3))

    return contours, volume_map


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

def export_all_volumes(ctrs, volume_map, data_path, overlay_path, crop_size=100, num_classes=4, num_slices=5, num_phase_in_cycle=20, is_all_valid_slice=True):
    print('\nProcessing {:d} volumes and labels ...\n'.format(len(ctrs)))
    volumes = np.zeros((len(ctrs), crop_size, crop_size, num_slices, 1))
    volume_masks = np.zeros((len(ctrs), crop_size, crop_size, num_slices, num_classes))
    idx = 0
    case = []
    img_no = []
    for i, contour in enumerate(ctrs):
        volume, volume_mask = read_volume(contour, volume_map, data_path, num_classes, num_slices, num_phase_in_cycle, crop_size, is_all_valid_slice=is_all_valid_slice)
        if len(volume) > 0:
            volumes[idx] = volume
            volume_masks[idx] = volume_mask
            case.append(contour.case)
            img_no.append(contour.img_no)
            idx = idx + 1

    volumes = volumes[0:idx-1]
    volume_masks = volume_masks[0:idx-1]
    return volumes, volume_masks, case, img_no

if __name__== '__main__':
    is_train = True
    contour_type = 'a'

    weight_path = None
    #weight_path = '.\\model_logs\\sunnybrook_a_unet_3d_Inv_e135_a8_f8_775_d4_s5_allvalid_mvn.h5'
    shuffle = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    crop_size = 128
    num_slices = 5
    num_phase_in_cycle = 20
    save_path = 'model_logs'
    verbosity = 1
    standard_weight = 1.0
    low_weight = 0.5
    hight_weight = 2.0

    patience = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
    early_stop = 50  # training will be stopped after this many epochs without the validation loss improving
    initial_learning_rate = 0.00001
    learning_rate_drop = 0.5  # factor by which the learning rate will be reduced

    print('Mapping ground truth contours to images in train...')
    train_ctrs, volume_map = map_all_contours(TRAIN_CONTOUR_PATH)
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(train_ctrs)

    print('Done mapping training set')
    num_classes = 3
    split = int(0.1*len(train_ctrs))
    dev_ctrs = train_ctrs[0:split]
    train_ctrs = train_ctrs[split:]
    
    print('\nBuilding Train dataset ...')
    img_train, mask_train, _, __ = export_all_volumes(train_ctrs,
                                                    volume_map,
                                                    TRAIN_IMG_PATH,
                                                    TRAIN_OVERLAY_PATH,
                                                    crop_size=crop_size,
                                                    num_classes=num_classes,
                                                    num_slices=num_slices,
                                                    num_phase_in_cycle=num_phase_in_cycle,
                                                    is_all_valid_slice=True)
    print('\nBuilding Dev dataset ...')
    img_dev, mask_dev, _, __ = export_all_volumes(dev_ctrs,
                                                   volume_map,
                                                   TRAIN_IMG_PATH,
                                                   TRAIN_OVERLAY_PATH,
                                                   crop_size=crop_size,
                                                   num_classes=num_classes,
                                                   num_slices=num_slices,
                                                   num_phase_in_cycle=num_phase_in_cycle,
                                                   is_all_valid_slice=True
                                           )
    
    input_shape = (crop_size, crop_size, num_slices, 1)

    kwargs = dict(
        rotation_range=90,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        data_format="channels_last",
        fill_mode='constant',
    )
    image_datagen = CardiacVolumeDataGenerator(**kwargs)
    mask_datagen = CardiacVolumeDataGenerator(**kwargs)
    aug_img_path = os.path.join(TRAIN_AUG_PATH, "Image")
    aug_mask_path = os.path.join(TRAIN_AUG_PATH, "Mask")
    img_train = image_datagen.fit(img_train, augment=True, seed=seed, rounds=8, toDir=None)
    mask_train = mask_datagen.fit(mask_train, augment=True, seed=seed, rounds=8, toDir=None)

    epochs = 200
    mini_batch_size = 1

    image_generator = image_datagen.flow(img_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    train_generator = zip_longest(image_generator, mask_generator)

    dev_generator = (img_dev, mask_dev)
    
    max_iter = int(np.ceil(len(img_train) / mini_batch_size)) * epochs
    steps_per_epoch = int(np.ceil(len(img_train) / mini_batch_size))
    curr_iter = 0

    callbacks = []
    # ####################### tfboard ###########################
    if K.backend() == 'tensorflow':
        tensorboard = TensorBoard(log_dir=os.path.join(save_path, 'logs_unet_3d_Inv'), histogram_freq=10, write_graph=False,
                                  write_grads=False, write_images=False)
        callbacks.append(tensorboard)
    # ################### checkpoint saver#######################
    callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=patience,
                                       verbose=verbosity))
    callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stop))
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'check_point_model.hdf5'),
                                 save_weights_only=False,
                                 save_best_only=False,
                                 period=20)  # .{epoch:d}
    callbacks.append(checkpoint)
    class_weight = dict([(i, low_weight) for i in range(num_classes)])
    class_weight[1] = hight_weight
    class_weight[2] = hight_weight

    if(is_train):
        if(weight_path == None):
            model = unet_model_3d_Inv(input_shape, pool_size=(2, 2, 1), kernel=(7, 7, 5), n_labels=3, initial_learning_rate=0.00001,
                              deconvolution=False, depth=4, n_base_filters=4, include_label_wise_dice_coefficients=True, batch_normalization=True, weights=None)
        else:
            model = resume_training(weight_path)

    else:
        model = unet_model_3d_Inv(input_shape, pool_size=(2, 2, 1), kernel=(7, 7, 5), n_labels=3, initial_learning_rate=0.00001,
                              deconvolution=False, depth=4, n_base_filters=4, include_label_wise_dice_coefficients=True, batch_normalization=True, weights=weight_path)

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=dev_generator,
                        validation_steps=img_dev.__len__(),
                        epochs=epochs,
                        callbacks=callbacks,
                        workers=1,
                        class_weight=None
                        )
    save_file = '_'.join(['sunnybrook', contour_type, 'unet', '3d']) + '.h5'
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
