import os
import sys

import numpy as np

from scipy.misc import imsave
import scipy.ndimage

import dicom as pydicom


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
TEST_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'Sunnybrook Cardiac MR Database ContoursPart2',
                   'ValidationDataContours')
TEST_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                   'Sunnybrook Cardiac MR Database DICOMPart2',
                    'ValidationDataDICOM')
TEST_OVERLAY_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
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

training_dicom_dir = TRAIN_IMG_PATH
training_labels_dir = TRAIN_CONTOUR_PATH
training_png_dir = "./Data/Training/Images/Sunnybrook_Part3"
training_png_labels_dir = "./Data/Training/Labels/Sunnybrook_Part3"
testing_dicom_dir = TEST_IMG_PATH
testing_labels_dir = TEST_CONTOUR_PATH
testing_png_dir = "./Data/Testing/Images/Sunnybrook_Part2"
testing_png_labels_dir = "./Data/Testing/Labels/Sunnybrook_Part2"
online_dicom_dir = ONLINE_IMG_PATH
online_labels_dir = ONLINE_CONTOUR_PATH
online_png_dir = "./Data/Online/Images/Sunnybrook_Part1"
online_png_labels_dir = "./Data/Online/Labels/Sunnybrook_Part1"

if not os.path.exists(training_png_dir):
    os.makedirs(training_png_dir)
if not os.path.exists(training_png_labels_dir):
    os.makedirs(training_png_labels_dir)
if not os.path.exists(testing_png_dir):
    os.makedirs(testing_png_dir)
if not os.path.exists(testing_png_labels_dir):
    os.makedirs(testing_png_labels_dir)
if not os.path.exists(online_png_dir):
    os.makedirs(online_png_dir)
if not os.path.exists(online_png_labels_dir):
    os.makedirs(online_png_labels_dir)
for labels_dir, dicom_dir, png_dir, png_labels_dir in [[training_labels_dir,training_dicom_dir, training_png_dir, training_png_labels_dir],
                                                       [testing_labels_dir,testing_dicom_dir, testing_png_dir, testing_png_labels_dir],
                                                       [online_labels_dir,online_dicom_dir, online_png_dir, online_png_labels_dir]]:
    for root, dirs, files in os.walk(labels_dir):
        for file in files:
            if file.endswith("-icontour-manual.txt"):
                try:
                    prefix, _ = os.path.split(root)
                    prefix, _ = os.path.split(prefix)
                    _, patient = os.path.split(prefix)

                    file_fn = file.strip("-icontour-manual.txt") + ".dcm"
                    print(file_fn)
                    print(patient)
                    dcm = pydicom.read_file(os.path.join(dicom_dir, patient, "DICOM", file_fn))
                    print(dcm.pixel_array.shape)
                    img = np.concatenate((dcm.pixel_array[...,None], dcm.pixel_array[...,None], dcm.pixel_array[...,None]), axis=2)
                    labels = np.zeros_like(dcm.pixel_array)

                    print(img.shape)
                    print(labels.shape)

                    with open(os.path.join(root, file)) as labels_f:
                        for line in labels_f:
                            x, y = line.split(" ")
                            labels[int(float(y)), int(float(x))] = 128

                    labels = scipy.ndimage.binary_fill_holes(labels)

                    img_labels = np.concatenate((labels[..., None], labels[..., None], labels[..., None]), axis=2)

                    imsave(os.path.join(png_dir, patient + "-" + file_fn + ".png"), img)
                    imsave(os.path.join(png_labels_dir, patient + "-" + file_fn + ".png"), img_labels)
                except Exception as e:
                    print(e)

