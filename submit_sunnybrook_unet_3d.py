#!/usr/bin/env python2.7

import re, sys, os
import shutil, cv2
import numpy as np

from train_sunnybrook_unet_3d import read_volume, map_all_contours, export_all_volumes, map_endo_contours
from helpers import reshape, get_SAX_SERIES, draw_result, draw_image_overlay
from unet_model_3d import unet_model_3d, dice_coef_endo_each, dice_coef_myo_each, resume_training
from CardiacImageDataGenerator import CardiacImageDataGenerator, CardiacVolumeDataGenerator
from unet_model_3d_Inv import unet_model_3d_Inv, resume_training

SAX_SERIES = get_SAX_SERIES()
SUNNYBROOK_ROOT_PATH = 'D:\cardiac_data\Sunnybrook'
VAL_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                                'Sunnybrook Cardiac MR Database ContoursPart2',
                                'ValidationDataContours')
VAL_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database DICOMPart2',
                            'ValidationDataDICOM')
VAL_OVERLAY_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                                'Sunnybrook Cardiac MR Database OverlayPart2',
                                'ValidationDataOverlay')
ONLINE_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                                   'Sunnybrook Cardiac MR Database ContoursPart1',
                                   'OnlineDataContours')
ONLINE_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                               'Sunnybrook Cardiac MR Database DICOMPart1',
                               'OnlineDataDICOM')
ONLINE_OVERLAY_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                                   'Sunnybrook Cardiac MR Database OverlayPart1',
                                   'OnlineDataOverlay')
TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart3',
                            'TrainingDataContours')
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook Cardiac MR Database DICOMPart3',
                              'TrainingDataDICOM')
TRAIN_OVERLAY_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook Cardiac MR Database OverlayPart3',
                              'TrainingOverlayImage')
SAVE_VAL_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                             'Sunnybrook_val_submission')
SAVE_ONLINE_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                                'Sunnybrook_online_submission')


def create_submission(contours, volume_map, data_path, output_path, num_slices, num_phase_in_cycle, contour_type='a', debug=False):
    if contour_type == 'a':
        weights = 'model_logs/sunnybrook_a_unet_3d.h5'
    else:
        sys.exit('\ncontour type "%s" not recognized\n' % contour_type)

    crop_size = 128
    input_shape = (crop_size, crop_size, num_slices, 1)
    num_classes = 3
    volumes, vol_masks, cases, img_nos = export_all_volumes(contours,
                                                            volume_map,
                                                            data_path,
                                                            output_path,
                                                            crop_size,
                                                            num_classes=num_classes,
                                                            num_slices=num_slices,
                                                            num_phase_in_cycle=num_phase_in_cycle,
                                                            is_all_valid_slice=True)

    model = unet_model_3d_Inv(input_shape, pool_size=(2, 2, 1), kernel=(7, 7, 5), n_labels=3, initial_learning_rate=0.00001,
                              deconvolution=False, depth=4, n_base_filters=4, include_label_wise_dice_coefficients=True, batch_normalization=True, weights=weights)

    if debug:
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
        seed = 1234
        np.random.seed(seed)
        image_datagen = CardiacVolumeDataGenerator(**kwargs)
        mask_datagen = CardiacVolumeDataGenerator(**kwargs)
        volumes = image_datagen.fit(volumes, augment=True, seed=seed, rounds=8, toDir=None)
        vol_masks = mask_datagen.fit(vol_masks, augment=True, seed=seed, rounds=8, toDir=None)
        result = model.evaluate(volumes, vol_masks, batch_size=8)
        result = np.round(result, decimals=10)
        print('\nResult {:s}:\n{:s}'.format(str(model.metrics_names), str(result)))
    else:
        pred_masks = model.predict(volumes, batch_size=8, verbose=1)
        print('\nEvaluating ...')
        result = model.evaluate(volumes, vol_masks, batch_size=8)
        result = np.round(result, decimals=10)
        print('\nResult {:s}:\n{:s}'.format(str(model.metrics_names), str(result)))
        num = 0

        for c_type in ['i', 'm']:
            for idx in range(len(volumes)):
                volume = volumes[idx]

                h, w, s, d = volume.shape
                for s_i in range(s):
                    img = volume[...,s_i, 0]
                    if c_type == 'i':
                        tmp = pred_masks[idx, ..., s_i, 2]
                    elif c_type == 'm':
                        tmp = pred_masks[idx, ..., s_i, 1]

                    tmp = tmp[..., np.newaxis]
                    tmp = reshape(tmp, to_shape=(h, w, d))
                    tmp = np.where(tmp > 0.5, 255, 0).astype('uint8')
                    tmp2, coords, hierarchy = cv2.findContours(tmp.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    if not coords:
                        print('\nNo detection in case: {:s}; image: {:d}'.format(cases[idx], img_nos[idx]))
                        coords = np.ones((1, 1, 1, 2), dtype='int')

                    overlay_full_path = os.path.join(save_dir, cases[idx], 'Overlay')
                    if not os.path.exists(overlay_full_path):
                        os.makedirs(overlay_full_path)
                    if 'Overlay' in overlay_full_path:
                        out_file = 'IM-0001-%s-%04d-%01d.png' % (c_type, img_nos[idx], s_i)
                        draw_image_overlay(img, out_file, overlay_full_path, c_type, coords)


            print('\nNumber of multiple detections: {:d}'.format(num))
            dst_eval = os.path.join(save_dir, 'evaluation_{:s}.txt'.format(c_type))
            with open(dst_eval, 'wb') as f:
                f.write(('Dev set result {:s}:\n{:s}'.format(str(model.metrics_names), str(result))).encode('utf-8'))
                f.close()

            # Detailed evaluation:
            detail_eval = os.path.join(save_dir, 'evaluation_detail_{:s}.csv'.format(c_type))
            evalEndoArr = []
            evalMyoArr = []
            resArr = [cases, img_nos]
            for s_i in range(s):
                resArr.append(list(dice_coef_endo_each(vol_masks[...,s_i,:], pred_masks[...,s_i,:])))
            for s_i in range(s):
                resArr.append(list(dice_coef_myo_each(vol_masks[..., s_i, :], pred_masks[..., s_i, :])))



            resArr = np.transpose(resArr)
            np.savetxt(detail_eval, resArr, fmt='%s', delimiter=',')


        # np.savetxt(f, '\nDev set result {:s}:\n{:s}'.format(str(model.metrics_names), str(result)))

if __name__ == '__main__':
    contour_type = 'a'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num_slices = 5
    num_phase_in_cycle = 20
    _debug = False

    if _debug:
        save_dir = 'D:\cardiac_data\Sunnybrook\Sunnybrook_debug_submission_unet_3d_Inv'
        print('\nProcessing online ' + contour_type + ' contours...')
        online_ctrs, volume_map = map_all_contours(TRAIN_CONTOUR_PATH)
        create_submission(online_ctrs, volume_map, TRAIN_IMG_PATH, TRAIN_OVERLAY_PATH, num_slices, num_phase_in_cycle,
                          contour_type, _debug)

    save_dir = 'D:\cardiac_data\Sunnybrook\Sunnybrook_online_submission_unet_3d_Inv'
    print('\nProcessing online ' + contour_type + ' contours...')
    online_ctrs, volume_map = map_all_contours(ONLINE_CONTOUR_PATH)
    create_submission(online_ctrs, volume_map, ONLINE_IMG_PATH, ONLINE_OVERLAY_PATH, num_slices, num_phase_in_cycle, contour_type, _debug)
    #create_endo_submission(online_endos, ONLINE_IMG_PATH, ONLINE_OVERLAY_PATH, contour_type)

    save_dir = 'D:\cardiac_data\Sunnybrook\Sunnybrook_val_submission_unet_3d_e135_a8_f8_775_d4_s5_allvalid_mvn'
    print('\nProcessing val ' + contour_type + ' contours...')
    val_ctrs, volume_map = map_all_contours(VAL_CONTOUR_PATH)
    create_submission(val_ctrs, volume_map, VAL_IMG_PATH, VAL_OVERLAY_PATH, num_slices, num_phase_in_cycle, contour_type, _debug)
    #create_endo_submission(val_endos, VAL_IMG_PATH, VAL_OVERLAY_PATH, contour_type)

    print('\nAll done.')

