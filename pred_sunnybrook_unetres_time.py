#!/usr/bin/env python2.7

import re, sys, os
import shutil, cv2
import numpy as np

from train_sunnybrook_unetres import read_contour, map_all_cases, export_all_contours, read_all_contour
from helpers import reshape, get_SAX_SERIES, draw_result, draw_image_overlay, center_crop, center_crop_3d

from unet_res_model_Inv import unet_res_model_Inv
from unet_model_time import unet_res_model_time, dice_coef, dice_coef_each
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
SAVE_VAL_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook_val_submission')
SAVE_ONLINE_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook_online_submission')

def create_submission(cases, data_path, output_path ,contour_type = 'i'):

    weight_t = 'model_logs/sunnybrook_a_unetres_inv_time.h5'
    weight_s = 'model_logs/sunnybrook_i_unetres_inv.h5'
    crop_size = 128
    num_phases = 3
    num_classes = 2

    input_shape = (num_phases, crop_size, crop_size, 1)
    input_shape_s = (crop_size, crop_size, 1)
    model_s = unet_res_model_Inv(input_shape, num_classes, nb_filters=16, transfer=True, contour_type=contour_type, weights=weight_s)
    model_t = unet_res_model_time(input_shape, num_classes, nb_filters=16, n_phases=num_phases, transfer=True, contour_type=contour_type, weights=weight_t)
    for idx, case in enumerate(cases):
        print('\nPredict image sequence {:d}'.format(idx))
        images, _, file_names = read_all_contour(case, data_path, num_classes)
        images_crop = center_crop_3d(images, crop_size=crop_size)
        pred_masks = model_s.predict(images_crop, batch_size=32, verbose=1)
        p, h, w, d = images.shape

        for idx in range(p):
            image = images[idx, ...]
            tmp = pred_masks[idx,:]
            out_file = file_names[idx]
            tmp = reshape(tmp, to_shape=(h, w, d))
            tmp = np.where(tmp > 0.5, 255, 0).astype('uint8')
            tmp2, coords, hierarchy = cv2.findContours(tmp.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if not coords:
                print('\nNo detection in case: {:s}; image: {:s}'.format(case, out_file))
                coords = np.ones((1, 1, 1, 2), dtype='int')

            output_full_path = os.path.join(output_path, case)

            p = re.compile("dcm")
            out_file = p.sub('jpg', out_file)
            draw_image_overlay(image, out_file, output_full_path, contour_type, coords)



if __name__== '__main__':
    contour_type = 'i'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    save_dir = 'D:\cardiac_data\Sunnybrook\Sunnybrook_val_submission_unetres_inv_time'
    print('\nProcessing val ' + contour_type + ' contours...')
    val_cases = list(map_all_cases(VAL_CONTOUR_PATH))
    create_submission(val_cases, VAL_IMG_PATH, save_dir, contour_type)

    save_dir = 'D:\cardiac_data\Sunnybrook\Sunnybrook_online_submission_unetres_inv_time'
    print('\nProcessing online '+contour_type+' contours...')
    online_cases = list(map_all_cases(ONLINE_CONTOUR_PATH))
    create_submission(online_cases, ONLINE_IMG_PATH, save_dir, contour_type)


    print('\nAll done.')

