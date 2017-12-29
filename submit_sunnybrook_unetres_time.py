#!/usr/bin/env python2.7

import re, sys, os
import shutil, cv2
import numpy as np
from keras import backend as K
from train_sunnybrook_unet_time import read_contour, export_all_contours, map_all_contours
from helpers import reshape, get_SAX_SERIES, draw_result, draw_image_overlay, center_crop, center_crop_3d
from scipy.misc import imsave
from unet_res_model_Inv import unet_res_model_Inv
from unet_model_time import unet_res_model_time, dice_coef
from metrics_common import dice_coef_each
SAX_SERIES = get_SAX_SERIES()
SUNNYBROOK_ROOT_PATH = 'D:\cardiac_data\Sunnybrook'
TEMP_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'Sunnybrook Cardiac MR Database Temp',
                   'Temp')
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
DEBUG_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart3',
                            'Debug')
DEBUG_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook Cardiac MR Database DICOMPart3',
                              'Debug')
DEBUG_OVERLAY_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook Cardiac MR Database OverlayPart3',
                              'Debug')

def create_submission(contours, data_path, output_path ,contour_type = 'i'):

    weight_t = 'model_logs/sunnybrook_a_unetres_inv_time.h5'
    weight_s = 'model_logs/sunnybrook_i_unetres_inv_drop_acdc.h5'
    crop_size = 128
    num_phases = 5
    num_classes = 2
    phase_dilation = 4
    input_shape = (num_phases, crop_size, crop_size, 1)
    input_shape_s = (crop_size, crop_size, 1)
    model_s = unet_res_model_Inv(input_shape_s, num_classes, nb_filters=8, transfer=True, contour_type=contour_type, weights=weight_s)
    model_t = unet_res_model_time(input_shape, num_classes, nb_filters=32, n_phases=num_phases, dilation=1, transfer=True, contour_type=contour_type, weights=weight_t)
    images, masks = export_all_contours(contours,
                                        data_path,
                                        output_path,
                                        crop_size=crop_size,
                                        num_classes=num_classes,
                                        num_phases=num_phases,
                                        phase_dilation=phase_dilation)
    s, p, h, w, d = images.shape
    print('\nFirst step predict set ...')
    temp_image_t = np.reshape(images, (s*p, h, w, d))
    temp_mask_t = model_s.predict(temp_image_t, batch_size=4, verbose=1)
    temp_mask_t = np.reshape(temp_mask_t, (s, p, h, w, d))
    # for idx_s in range(s):
        # img_t = images[idx_s,...]
        # temp_mask_t[idx_s] = model_s.predict(img_t)

        # for idx_p in range(p):
        #     mask = temp_mask_t[idx_s, idx_p, ...]
        #     img = images[idx_s, idx_p, ...]
        #     img = np.squeeze(img*mask)
        #     img_name = '{:d}-{:d}'.format(idx_s, idx_p)
        #     imsave(os.path.join(TEMP_CONTOUR_PATH, img_name + ".png"), img)

    print('\nTotal sample is {:d} for 2nd evaluation.'.format(s))
    print('\nSecond step predict set ...')
    pred_masks = model_t.predict(temp_mask_t, batch_size=4, verbose=1)
    print('\nEvaluating dev set ...')
    result = model_t.evaluate(temp_mask_t, masks, batch_size=4)
    result = np.round(result, decimals=10)
    print('\nDev set result {:s}:\n{:s}'.format(str(model_t.metrics_names), str(result)))

    for idx, ctr in enumerate(contours):
        print('\nPredict image sequence {:d}'.format(idx))
        img, mask = read_contour(ctr, data_path, num_classes, num_phases, num_phases_in_cycle=20, phase_dilation=phase_dilation)
        p, h, w, d = img.shape
        tmp = np.squeeze(pred_masks[idx, :])
        if tmp.ndim == 2:
            tmp = tmp[:,:,np.newaxis]
        tmp = np.where(tmp > 0.5, 255, 0).astype('uint8')
        tmp2, coords, hierarchy = cv2.findContours(tmp.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if not coords:
            print('\nNo detection in case: {:s}; image: {:d}'.format(ctr.case, ctr.img_no))
            coords = np.ones((1, 1, 1, 2), dtype='int')

        overlay_full_path = os.path.join(save_dir, ctr.case, 'Overlay')
        draw_result(ctr, data_path, overlay_full_path, contour_type, coords)

    dst_eval = os.path.join(save_dir, 'evaluation_{:s}.txt'.format(contour_type))
    with open(dst_eval, 'wb') as f:
        f.write(('Dev set result {:s}:\n{:s}'.format(str(model_t.metrics_names), str(result))).encode('utf-8'))
        f.close()

    # Detailed evaluation:
    masks = np.squeeze(masks)
    pred_masks = np.squeeze(pred_masks)
    detail_eval = os.path.join(save_dir, 'evaluation_detail_{:s}.csv'.format(contour_type))
    evalArr = dice_coef_each(masks, pred_masks)
    caseArr = [ctr.case for ctr in contours]
    imgArr = [ctr.img_no for ctr in contours]
    resArr = [caseArr, imgArr]
    resArr.append(list(evalArr))
    resArr = np.transpose(resArr)
    np.savetxt(detail_eval, resArr, fmt='%s', delimiter=',')


if __name__== '__main__':
    contour_type = 'i'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    save_dir = 'D:\cardiac_data\Sunnybrook\Sunnybrook_val_submission_unetres_time_acdc_p5_a4_e30'
    print('\nProcessing val ' + contour_type + ' contours...')
    val_ctrs = list(map_all_contours(VAL_CONTOUR_PATH))
    create_submission(val_ctrs, VAL_IMG_PATH, save_dir, contour_type)

    save_dir = 'D:\cardiac_data\Sunnybrook\Sunnybrook_online_submission_unetres_time_acdc_p5_a4_e30'
    print('\nProcessing online '+contour_type+' contours...')
    online_ctrs = list(map_all_contours(ONLINE_CONTOUR_PATH))
    create_submission(online_ctrs, ONLINE_IMG_PATH, save_dir, contour_type)


    print('\nAll done.')

