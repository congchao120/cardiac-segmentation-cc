import os, re
import random

import numpy as np
from scipy.misc import imsave
import scipy.misc
from keras import backend as K
class DataIOProc():
    def __init__(self, data_dir, study_case):
        self.data_dir = data_dir
        self.study_case = study_case


    def save_image_4d(self, data_4d, sub_dir):
        save_path = os.path.join(self.data_dir, self.study_case, sub_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        s, p, h, w, d = data_4d.shape
        if d != 1:
            print("The last dimension of data should be 1!")
            return

        for idx_s in range(s):
            for idx_p in range(p):
                img = data_4d[idx_s, idx_p, ...]
                img = np.squeeze(img)
                img_name = '{:d}-{:d}'.format(idx_s, idx_p)
                imsave(os.path.join(save_path, img_name + ".png"), img)


    def load_image_4d(self, sub_dir, s, p, h, w, d):
        save_path = os.path.join(self.data_dir, self.study_case, sub_dir)
        if not os.path.exists(save_path):
            print("No data!")
            return

        if d != 1:
            print("The last dimension of data should be 1!")
            return

        data_4d = np.zeros((s, p, w, h, d), dtype=K.floatx())

        for label_root, dir, files in os.walk(save_path):
            for file in files:
                if not file.endswith((".png")):
                    continue

                try:
                    image = scipy.misc.imread(os.path.join(save_path, file))
                    image = image.astype('float32')/255.0
                    image = image[..., np.newaxis]
                    match = re.search(r'(\d)-(\d).*', file)
                    s = int(match.group(1))
                    p = int(match.group(2))
                    data_4d[s, p, ...] = image
                except Exception as e:
                    print(e)
        return data_4d

    def save_data_4d(self, data_4d, save_name):
        save_path = os.path.join(self.data_dir, self.study_case)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_file = os.path.join(save_path,save_name)
        data_4d.tofile(save_file)


    def load_data_4d(self, load_name, s, p, h, w, d):
        save_path = os.path.join(self.data_dir, self.study_case)
        if not os.path.exists(save_path):
            print("No data!")
            return

        save_file = os.path.join(save_path, load_name)
        data_4d = np.fromfile(save_file, dtype='float32')
        data_4d = np.reshape(data_4d, [s, p, h, w, d])
        return data_4d