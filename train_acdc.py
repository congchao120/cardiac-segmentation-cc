#!/usr/bin/env python2.7

import dicom, cv2, re
import os, fnmatch, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from itertools import zip_longest
from LearningRate_Batch import LearningRateBatchScheduler
from keras.callbacks import *
from fcn_model_multiclass import fcn_model_multiclass
from fcn_model_resnet50 import fcn_model_resnet50
from helpers import center_crop, lr_poly_decay
from metrics_acdc import load_nii
import pylab
import matplotlib.pyplot as plt
from CardiacImageDataGenerator import CardiacImageDataGenerator
from CardiacTensorBoard import CardiacTensorBoard

seed = 1234
np.random.seed(seed)

RVSC_ROOT_PATH = 'D:\cardiac_data\ACDC'

TRAIN_PATH = os.path.join(RVSC_ROOT_PATH, 'training')
DEBUG_PATH = os.path.join(RVSC_ROOT_PATH, 'debug')
TRAIN_OVERLAY_PATH = os.path.join(RVSC_ROOT_PATH, 'overlay')
class VolumeCtr(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'patient(\d{03})_frame(\d{02})*', ctr_path)
        self.patient_no = match.group(1)
        self.img_no = match.group(2)
        gt, _, header = load_nii(ctr_path)
        self.total_number = gt.shape[2]

# ###############learning rate scheduler####################
def lr_scheduler(curr_epoch, curr_iter):
    total_iter = curr_epoch*steps_per_epoch + curr_iter
    lrate = lr_poly_decay(model, base_lr, total_iter, max_iter, power=0.5)

    print(' - lr: %f' % lrate)
    return lrate

def read_contour(contour, data_path, return_mask=True, type="i"):
    img_path = os.path.join(data_path, 'patient{:s}'.format(contour.patient_no))

    image_name = 'patient{:s}_frame{:s}.nii.gz'.format(contour.patient_no, contour.img_no)
    gt_name = 'patient{:s}_frame{:s}_gt.nii.gz'.format(contour.patient_no, contour.img_no)
    full_image_path = os.path.join(img_path, image_name)
    full_gt_path = os.path.join(img_path, gt_name)
    volume, _, header = load_nii(full_image_path)
    volume_gt, _, header = load_nii(full_gt_path)
    volume = volume.astype('int')
    if type == "i":
        volume_gt = np.where(volume_gt == 3, 1, 0).astype('uint8')
    elif type == "o":
        volume_gt = np.where(volume_gt >= 2, 1, 0).astype('uint8')
    elif type == "r":
        volume_gt = np.where(volume_gt == 1, 1, 0).astype('uint8')
    elif type == "a":
        volume_gt = volume_gt.astype('uint8')

    if not return_mask:
        return volume, None

    return volume, volume_gt

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


def export_all_contours(contours, data_path, overlay_path, crop_size, contour_type):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    total_number = 0
    for volume_ctr in contours:
        total_number += volume_ctr.total_number

    images = np.zeros((total_number, crop_size, crop_size, 1))
    masks = np.zeros((total_number, crop_size, crop_size, num_classes))
    idx = 0
    for contour in contours:
        vol, vol_mask = read_contour(contour, data_path, return_mask=True, type=contour_type)
        #draw_contour(contour, data_path, overlay_path, type=contour_type)
        for i in range(0, vol.shape[2]):
            img = vol[:,:,i]
            mask = vol_mask[:,:,i]
            img = np.swapaxes(img, 0, 1)
            mask = np.swapaxes(mask, 0, 1)

            if img.ndim < 3:
                img = img[..., np.newaxis]
            if mask.ndim < 3:
                if contour_type != "a":
                    mask = mask[..., np.newaxis]
                elif contour_type == "a":
                    h, w = mask.shape
                    classify = np.zeros((h, w, num_classes), dtype="uint8")
                    classify[..., 1] = np.where(mask == 1, 1, 0)
                    classify[..., 2] = np.where(mask == 2, 1, 0)
                    classify[..., 3] = np.where(mask == 3, 1, 0)
                    classify[..., 0] = np.where(mask == 0, 1, 0)
                    mask = classify
            img = center_crop(img, crop_size=crop_size)
            mask = center_crop(mask, crop_size=crop_size)
            images[idx] = img
            masks[idx] = mask
            idx = idx + 1
    return images, masks


if __name__== '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: python %s <i/o> <gpu_id>' % sys.argv[0])
    contour_type = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

    crop_size = 100
    #weight_path = 'C:\\Users\\congchao\\PycharmProjects\\cardiac-segmentation-master\\model_logs\\acdc_weights.hdf5'
    weight_path = None
    save_path = 'model_logs'
    num_classes = 4
    print('Mapping ground truth '+contour_type+' contours to images in train...')
    train_ctrs = list(map_all_contours(DEBUG_PATH, contour_type, shuffle=False))
    print('Done mapping training set')
    split = int(0.1*len(train_ctrs))
    dev_ctrs = train_ctrs[0:split]
    train_ctrs = train_ctrs[split:]
    print('\nBuilding train dataset ...')
    global img_train
    global mask_train
    img_train, mask_train = export_all_contours(train_ctrs,
                                                DEBUG_PATH,
                                                TRAIN_OVERLAY_PATH,
                                                crop_size=crop_size,
                                                contour_type = contour_type)
    print('\nBuilding dev dataset ...')
    img_dev, mask_dev = export_all_contours(dev_ctrs,
                                            DEBUG_PATH,
                                            TRAIN_OVERLAY_PATH,
                                            crop_size=crop_size,
                                            contour_type = contour_type)

    input_shape = (crop_size, crop_size, 1)

    model = fcn_model_resnet50(input_shape, num_classes, transfer=True, contour_type='i', weights=None)

    kwargs = dict(
        rotation_range=180,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        data_format="channels_last",
    )
    '''
    kwargs = dict(
        horizontal_flip=True,
        vertical_flip=True,
        data_format="channels_last",
    )
    '''
    image_datagen = CardiacImageDataGenerator(**kwargs)
    mask_datagen = CardiacImageDataGenerator(**kwargs)
    img_train = image_datagen.fit(img_train, augment=True, seed=seed, rounds=1)
    mask_train = mask_datagen.fit(mask_train, augment=True, seed=seed, rounds=1)

    epochs = 80
    mini_batch_size = 1

    image_generator = image_datagen.flow(img_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    mask_generator = mask_datagen.flow(mask_train, shuffle=False,
                                    batch_size=mini_batch_size, seed=seed)
    train_generator = zip_longest(image_generator, mask_generator)
    dev_generator = (img_dev, mask_dev)

    max_iter = int(np.ceil(len(img_train)  / mini_batch_size)) * epochs
    steps_per_epoch = int(np.ceil(len(img_train)  / mini_batch_size))
    curr_iter = 0
    base_lr = K.eval(model.optimizer.lr)
    lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)
    scheduler = LearningRateBatchScheduler(lr_scheduler)
    # scheduler = LearningRateScheduler(lr_scheduler)
    callbacks = [scheduler]

    # ####################### tfboard ###########################
    if K.backend() == 'tensorflow':
        tensorboard = CardiacTensorBoard(log_dir=os.path.join(save_path, 'logs'), histogram_freq=1, write_graph=False,
                                  write_grads=False, write_images=False)
        callbacks.append(tensorboard)
    # ################### checkpoint saver#######################
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'acdc_weights.hdf5'),
                                 save_weights_only=True,
                                 save_best_only=True)  # .{epoch:d}
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
    save_file = '_'.join(['acdc', contour_type]) + '.h5'
    save_file = os.path.join(save_path, save_file)
    model.save_weights(save_file)



