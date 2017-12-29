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
from metrics_acdc import load_nii
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
TRAIN_AUG_PATH = os.path.join(ACDC_ROOT_PATH,
                        'Augmentation')
TRAIN_PATH = os.path.join(ACDC_ROOT_PATH, 'training')
DEBUG_PATH = os.path.join(ACDC_ROOT_PATH, 'debug')
TRAIN_OVERLAY_PATH = os.path.join(ACDC_ROOT_PATH, 'overlay')
TEMP_CONTOUR_PATH = os.path.join(ACDC_ROOT_PATH,
                   'ACDC Cardiac MR Database Temp',
                   'Temp')
class VolumeCtr(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'patient(\d{03})_frame(\d{02})*', ctr_path)
        self.patient_no = match.group(1)
        self.img_no = match.group(2)
        gt, _, header = load_nii(ctr_path)
        self.total_number = gt.shape[2]

def read_contour(contour, data_path, num_phases, num_phases_in_cycle, phase_dilation, contour_type='i', return_mask=True):
    img_path = os.path.join(data_path, 'patient{:s}'.format(contour.patient_no))

    image_name = 'patient{:s}_frame{:s}.nii.gz'.format(contour.patient_no, contour.img_no)
    gt_name = 'patient{:s}_frame{:s}_gt.nii.gz'.format(contour.patient_no, contour.img_no)
    full_image_path = os.path.join(img_path, image_name)
    full_gt_path = os.path.join(img_path, gt_name)
    volume, _, header = load_nii(full_image_path)
    volume_gt, _, header = load_nii(full_gt_path)
    volume = volume.astype('int')
    if contour_type == "i":
        volume_gt = np.where(volume_gt == 3, 1, 0).astype('uint8')
    elif contour_type == "o":
        volume_gt = np.where(volume_gt >= 2, 1, 0).astype('uint8')
    elif contour_type == "r":
        volume_gt = np.where(volume_gt == 1, 1, 0).astype('uint8')
    elif contour_type == "a":
        volume_gt = volume_gt.astype('uint8')

    volume_arr = find_neighbor_volumes(contour, data_path, num_phases, num_phases_in_cycle, phase_dilation)

    if volume_arr.ndim < 5:
        volume_arr = volume_arr[..., np.newaxis]
    if volume_gt.ndim < 4:
        volume_gt = volume_gt[np.newaxis, :, :, :, np.newaxis]

    if not return_mask:
        return volume_arr, None

    return volume_arr, volume_gt

def find_neighbor_volumes(contour, data_path, num_phases, num_phases_in_cycle, phase_dilation):
    volume_path = os.path.join(data_path, 'patient{:s}'.format(contour.patient_no))
    volume_name = 'patient{:s}_4d.nii.gz'.format(contour.patient_no)
    center_index = float(contour.img_no)
    full_volume_path = os.path.join(volume_path, volume_name)
    volume, _, header = load_nii(full_volume_path)
    volume = volume.astype('int')

    h, w, s, p = volume.shape
    phase_dilation = phase_dilation*p/num_phases_in_cycle

    volume_arr = np.zeros((num_phases, h, w, s), dtype="int")
    for i in range (num_phases):
        idx = int(center_index + (i - int(num_phases/2))*phase_dilation)%p
        volume_arr[i, ...] = volume[...,idx]


    return volume_arr

def draw_contour(contour, data_path, out_path, type="i", coords = []):
    img_path = os.path.join(data_path, 'patient{:s}'.format(contour.patient_no))
    image_name = 'patient{:s}_frame{:s}.nii.gz'.format(contour.patient_no, contour.img_no)
    gt_name = 'patient{:s}_frame{:s}_gt.nii.gz'.format(contour.patient_no, contour.img_no)
    full_image_path = os.path.join(img_path, image_name)
    full_gt_path = os.path.join(img_path, gt_name)

    volume, _, header = load_nii(full_image_path)
    volume_gt, _, header = load_nii(full_gt_path)
    img_size = volume.shape
    for i in range(0, img_size[2]):
        overlay_name = 'patient{:s}_frame{:s}_{:2d}_{:s}.png'.format(contour.patient_no, contour.img_no, i, type)
        full_overlay_path = os.path.join(img_path, overlay_name)
        if type != "a":
            img = volume[:, :, i]
            mask = volume_gt[:, :, i]
            img = np.swapaxes(img, 0, 1)
            mask = np.swapaxes(mask, 0, 1)
            if type == "i":
                mask = np.where(mask == 3, 255, 0).astype('uint8')
            elif type == "o":
                mask = np.where(mask >= 2, 255, 0).astype('uint8')
            elif type == "r":
                mask = np.where(mask == 1, 255, 0).astype('uint8')

            img = img.astype('int')

            tmp2, coords, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            if not coords:
                print('\nNo detection: {:s}, {:2d}'.format(contour.ctr_path, i))
                coords = np.ones((1, 1, 1, 2), dtype='int')
            if len(coords) > 1:
                print('\nMultiple detections: {:s}, {:2d}'.format(contour.ctr_path, i))
                lengths = []
                for coord in coords:
                    lengths.append(len(coord))
                coords = [coords[np.argmax(lengths)]]

            coords = np.squeeze(coords)

            if coords.ndim == 1:
                x, y = coords
            else:
                x, y = zip(*coords)

            plt.cla()
            pylab.imshow(img, cmap=pylab.cm.bone)

            if type == "i":
                plt.plot(x, y, 'r.')
            elif type == "o":
                plt.plot(x, y, 'b.')
            elif type == "r":
                plt.plot(x, y, 'g.')

        elif type == "a":
            img = volume[:, :, i]
            img = np.swapaxes(img, 0, 1)
            mask_i = volume_gt[:, :, i]
            mask_o = volume_gt[:, :, i]
            mask_r = volume_gt[:, :, i]
            mask_i = np.swapaxes(mask_i, 0, 1)
            mask_o = np.swapaxes(mask_o, 0, 1)
            mask_r = np.swapaxes(mask_r, 0, 1)

            mask_i = np.where(mask_i == 3, 255, 0).astype('uint8')
            mask_o = np.where(mask_o >= 2, 255, 0).astype('uint8')
            mask_r = np.where(mask_r == 1, 255, 0).astype('uint8')

            img = img.astype('int')

            tmp2, coords_i, hierarchy = cv2.findContours(mask_i.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            tmp2, coords_o, hierarchy = cv2.findContours(mask_o.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            tmp2, coords_r, hierarchy = cv2.findContours(mask_r.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            if not coords_i:
                print('\nNo detection endo: {:s}, {:2d}'.format(contour.ctr_path, i))
                coords_i = np.ones((1, 1, 1, 2), dtype='int')
            if len(coords_i) > 1:
                print('\nMultiple detections endo: {:s}, {:2d}'.format(contour.ctr_path, i))
                lengths = []
                for coord in coords_i:
                    lengths.append(len(coord))

                coords_i = [coords_i[np.argmax(lengths)]]

            coords_i = np.squeeze(coords_i)

            if not coords_o:
                print('\nNo detection epi: {:s}, {:2d}'.format(contour.ctr_path, i))
                coords_o = np.ones((1, 1, 1, 2), dtype='int')
            if len(coords_o) > 1:
                print('\nMultiple detections epi: {:s}, {:2d}'.format(contour.ctr_path, i))
                lengths = []
                for coord in coords_o:
                    lengths.append(len(coord))

                coords_o = [coords_o[np.argmax(lengths)]]

            coords_o = np.squeeze(coords_o)

            if not coords_r:
                print('\nNo detection right ventricle: {:s}, {:2d}'.format(contour.ctr_path, i))
                coords_r = np.ones((1, 1, 1, 2), dtype='int')
            if len(coords_r) > 1:
                print('\nMultiple detections right ventricle: {:s}, {:2d}'.format(contour.ctr_path, i))
                lengths = []
                for coord in coords_r:
                    lengths.append(len(coord))

                coords_r = [coords_r[np.argmax(lengths)]]

            coords_r = np.squeeze(coords_r)

            if coords_i.ndim == 1:
                x, y = coords_i
            else:
                x, y = zip(*coords_i)

            plt.cla()
            pylab.imshow(img, cmap=pylab.cm.bone)
            plt.plot(x, y, 'r.')

            if coords_o.ndim == 1:
                x, y = coords_o
            else:
                x, y = zip(*coords_o)

            plt.plot(x, y, 'b.')

            if coords_r.ndim == 1:
                x, y = coords_r
            else:
                x, y = zip(*coords_r)

            plt.plot(x, y, 'g.')

        plt.xlim(25, img.shape[1]-25)
        plt.ylim(25, img.shape[0]-25)
        pylab.savefig(full_overlay_path,bbox_inches='tight',dpi=200)

    #pylab.show()
    return

def map_all_contours(data_path, contour_type, shuffle=True):
    contours = [os.path.join(dirpath, f)
                for dirpath, dirnames, files in os.walk(data_path)
                for f in fnmatch.filter(files,
                                        'patient*'+ '_frame*_gt.*')]
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
    print('Number of examples: {:d}'.format(len(contours)))


    contours = map(VolumeCtr, contours)

    return contours


def export_all_contours(contours, data_path, overlay_path, crop_size, contour_type, num_classes=4, num_phases=5, phase_dilation=1, num_phases_in_cycle=30):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    if num_classes == 2:
        num_classes = 1
    total_number = 0
    for volume_ctr in contours:
        total_number += volume_ctr.total_number

    images = np.zeros((total_number, num_phases, crop_size, crop_size, 1))
    masks = np.zeros((total_number, 1, crop_size, crop_size, num_classes))
    idx = 0
    for contour in contours:
        vol, vol_mask = read_contour(contour, data_path, num_phases, num_phases_in_cycle, phase_dilation, contour_type=contour_type, return_mask=True)
        #draw_contour(contour, data_path, overlay_path, type=contour_type)
        p, w, h, s, d = vol.shape
        for i in range(0, s):
            img = vol[:,:,:,i,:]
            mask = vol_mask[:,:,:,i,:]
            img = np.swapaxes(img, 1, 2)
            mask = np.swapaxes(mask, 1, 2)

            img = center_crop_3d(img, crop_size=crop_size)
            mask = center_crop_3d(mask, crop_size=crop_size)
            images[idx] = img
            masks[idx] = mask
            idx = idx + 1
    return images, masks

if __name__== '__main__':

    contour_type = 'i'
    weight_s = 'model_logs/sunnybrook_i_unetres_inv_drop_acdc.h5'
    shuffle = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    crop_size = 128
    num_phases = 5
    save_path = 'model_logs'
    phase_dilation = 4
    num_phases_in_cycle = 30
    data_proc = DataIOProc(TEMP_CONTOUR_PATH, 'p5_a4')

    print('Mapping ground truth contours to images in train...')
    train_ctrs = list(map_all_contours(TRAIN_PATH, contour_type, shuffle=False))
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
                                                TRAIN_PATH,
                                                TRAIN_OVERLAY_PATH,
                                                contour_type = contour_type,
                                                crop_size=crop_size,
                                                num_classes=num_classes,
                                                num_phases=num_phases,
                                                phase_dilation=phase_dilation,
                                                num_phases_in_cycle=num_phases_in_cycle)
    print('\nBuilding Dev dataset ...')
    img_dev, mask_dev = export_all_contours(dev_ctrs,
                                            TRAIN_PATH,
                                            TRAIN_OVERLAY_PATH,
                                            contour_type=contour_type,
                                            crop_size=crop_size,
                                            num_classes=num_classes,
                                            num_phases=num_phases,
                                            phase_dilation=phase_dilation,
                                            num_phases_in_cycle = num_phases_in_cycle)


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
    img_train = image_datagen.fit(img_train, augment=True, seed=seed, rounds=4, toDir=None)
    mask_train = mask_datagen.fit(mask_train, augment=True, seed=seed, rounds=4, toDir=None)
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
