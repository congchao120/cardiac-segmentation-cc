import tensorflow as tf
import re, sys, os
import shutil, cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt
from helpers import reshape

SUNNYBROOK_ROOT_PATH = 'D:\cardiac_data\Sunnybrook'
SAVE_VAL_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook_val_submission')
SAVE_ONLINE_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'Sunnybrook_online_submission')

def draw_contour(image, image_name, out_path, contour_type='i', coords=None):
    out_full_name = os.path.join(out_path, image_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    image = image[..., 0]
    img_size = image.shape
    plt.cla()
    pylab.imshow(image, cmap=pylab.cm.bone)

    if isinstance(coords, np.ndarray):
        if coords.ndim == 1:
            x, y = coords
        else:
            x, y = zip(*coords)

        if contour_type == 'i':
            plt.plot(x, y, 'r.')
        elif contour_type == 'o':
            plt.plot(x, y, 'b.')


        plt.xlim(50, img_size[0]-50)
        plt.ylim(50, img_size[1]-50)
        pylab.savefig(out_full_name,bbox_inches='tight',dpi=200)

    #pylab.show()
    return
def add_output_images(images, logits, labels, max_outputs=3):

    tf.summary.image('input', images, max_outputs=max_outputs)

    output_image_bw = images[..., 0]

    labels1 = tf.cast(labels[...,0], tf.float32)

    input_labels_image_r = labels1 + (output_image_bw * (1-labels1))
    input_labels_image = tf.stack([input_labels_image_r, output_image_bw, output_image_bw], axis=3)
    tf.summary.image('input_labels_mixed', input_labels_image, max_outputs=3)

    img_shape = tf.shape(images)
    classification1 = tf.image.resize_image_with_crop_or_pad(logits, img_shape[1], img_shape[2])[...,1]
    output_labels_image_r = classification1 + (output_image_bw * (1-classification1))

    output_labels_image = tf.stack([output_labels_image_r, output_image_bw, output_image_bw], axis=3)
    tf.summary.image('output_labels_mixed', output_labels_image, max_outputs=3)

    return

def save_output_images(images, logits, image_names, contour_type):
    save_dir = 'D:\cardiac_data\Sunnybrook\Sunnybrook_online_submission'
    overlay_full_path = os.path.join(save_dir, 'Overlay')
    img_shape = images.shape
    for idx in range(img_shape[0]):
        image = images[idx,...]
        image_name = image_names[idx]
        logit = logits[idx, ..., 1]
        logit = logit[..., np.newaxis]
        logit = reshape(logit, to_shape=(img_shape[1], img_shape[2], img_shape[3]))
        logit = np.where(logit > 0.5, 255, 0).astype('uint8')
        tmp2, coords, hierarchy = cv2.findContours(logit.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not coords:
            print('\nNo detection in image: {:s}'.format(image_name))
            coords = np.ones((1, 1, 1, 2), dtype='int')
        if len(coords) > 1:
            print('\nMultiple detections in image: {:s}'.format(image_name))
            # cv2.imwrite(data_path + '\\multiple_dets\\'+contour_type+'{:04d}.png'.format(idx), tmp)
            lengths = []
            for coord in coords:
                lengths.append(len(coord))
            coords = [coords[np.argmax(lengths)]]

        coords = np.squeeze(coords)
        draw_contour(image, image_name, overlay_full_path, contour_type, coords)


def save_output_eval(accuracy, image_names, contour_type):

    img_shape = image_names.shape
    resArr = []

    for idx in range(img_shape[0]):
        eval = accuracy[idx]
        img = image_names[idx]
        resArr = [resArr, np.transpose([img, eval])]

    return resArr

