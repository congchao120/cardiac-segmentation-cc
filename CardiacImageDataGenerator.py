from keras.preprocessing.image import ImageDataGenerator, transform_matrix_offset_center, apply_transform, random_channel_shift, flip_axis
from keras.preprocessing.image import Iterator as Iterator
from keras import backend as K
import numpy as np
import warnings
import os
from scipy import linalg
import pylab
import matplotlib.pyplot as plt


class ImageArrayIterator(Iterator):
    """Iterator yielding data from image array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(x, dtype=K.floatx())

        if self.x.ndim != 5:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 4 if data_format == 'channels_last' else 2
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            warnings.warn('NumpyArrayIterator is set to use the '
                          'data format convention "' + data_format + '" '
                                                                     '(channels on axis ' + str(
                channels_axis) + '), i.e. expected '
                                 'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                                                                                             'However, it was passed an array with shape ' + str(
                self.x.shape) +
                          ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(ImageArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            #x = self.image_data_generator.random_transform_array(x.astype(K.floatx()))
            #x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y

class CardiacImageDataGenerator(ImageDataGenerator):
    #Customized data augmentation method.

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None,
            toDir=None):
        """Fits internal statistics to some sample data.

        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' + self.data_format + '" '
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        if toDir != None:
            if not os.path.exists(toDir):
                os.makedirs(toDir)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
                    if toDir != None:
                        filename = 'img-%d.png' % (i + r * x.shape[0])
                        out_full_name = os.path.join(toDir, filename)
                        shape = ax.shape
                        if shape[3] == 1:
                            img = ax[i, ..., 0]
                            plt.cla()
                            pylab.imshow(img, cmap=pylab.cm.bone)
                            pylab.savefig(out_full_name, bbox_inches='tight')
                        elif shape[3] == 4:
                            img = ax[i, ..., 1:4]
                            plt.cla()
                            pylab.imshow(img)
                            pylab.savefig(out_full_name, bbox_inches='tight')

            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + self.zca_epsilon))), u.T)

        return x

    def fit_3d(self, x,
            augment=False,
            rounds=1,
            seed=None,
            toDir=None):
        """Fits internal statistics to some sample data.

        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            x: Numpy array, the data to fit on. Should have rank 5.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        # Raises
            ValueError: in case of invalid input `x`.
        """

        data_format = self.data_format

        if data_format == 'channels_first':
            self.channel_axis = 2
            self.phase_axis = 1
            self.row_axis = 3
            self.col_axis = 4
        if data_format == 'channels_last':
            self.channel_axis = 4
            self.phase_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 5:
            raise ValueError('Input to `.fit()` should have rank 5. '
                             'Got array with shape: ' + str(x.shape))

        if seed is not None:
            np.random.seed(seed)

        if toDir != None:
            if not os.path.exists(toDir):
                os.makedirs(toDir)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform_array(x[i])
                    if toDir != None:
                        for j in range(x.shape[1]):
                            filename = 'img-%d-%d.png' % (i + r * x.shape[0], j)
                            out_full_name = os.path.join(toDir, filename)
                            shape = ax.shape
                            if shape[4] == 1:
                                img = ax[i, j, ..., 0]
                                plt.cla()
                                pylab.imshow(img, cmap=pylab.cm.bone)
                                pylab.savefig(out_full_name, bbox_inches='tight')
                            elif shape[4] == 4:
                                img = ax[i, j, ..., 1:4]
                                plt.cla()
                                pylab.imshow(img)
                                pylab.savefig(out_full_name, bbox_inches='tight')
                            elif shape[4] == 3:
                                img = ax[i, j, ..., :]
                                plt.cla()
                                pylab.imshow(img)
                                pylab.savefig(out_full_name, bbox_inches='tight')

            x = ax


        return x

    def fit_to_directory(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Fits internal statistics to some sample data.

        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' + self.data_format + '" '
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + self.zca_epsilon))), u.T)

        return x


    def random_transform_array(self, x, seed=None):
        """Randomly augment a image array tensor.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_phase_axis = self.phase_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        p, h, w = x.shape[img_phase_axis], x.shape[img_row_axis], x.shape[img_col_axis]
        if transform_matrix is not None:
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            for i in range(p):
                x[i] = apply_transform(x[i], transform_matrix, img_channel_axis-1,
                                    fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            for i in range(p):
                x[i] = random_channel_shift(x[i],
                                         self.channel_shift_range,
                                         img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                for i in range(p):
                    x[i] = flip_axis(x[i], img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                for i in range(p):
                    x[i] = flip_axis(x[i], img_row_axis)

        return x

    def flow3d(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return ImageArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

class CardiacVolumeDataGenerator(ImageDataGenerator):
    #Customized data augmentation method.

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None,
            toDir=None):
        """Fits internal statistics to some sample data.

        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            x: Numpy array, the data to fit on. Should have rank 5.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        # Raises
            ValueError: in case of invalid input `x`.
        """

        data_format = self.data_format

        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
            self.slice_axis = 4
        if data_format == 'channels_last':
            self.channel_axis = 4
            self.row_axis = 1
            self.col_axis = 2
            self.slice_axis = 3
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 5:
            raise ValueError('Input to `.fit()` should have rank 5. '
                             'Got array with shape: ' + str(x.shape))

        if seed is not None:
            np.random.seed(seed)

        if toDir != None:
            if not os.path.exists(toDir):
                os.makedirs(toDir)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform_array(x[i])
                    if toDir != None:
                        for j in range(x.shape[self.slice_axis]):
                            filename = 'img-%d-%d.png' % (i + r * x.shape[0], j)
                            out_full_name = os.path.join(toDir, filename)
                            shape = ax.shape
                            if shape[self.channel_axis] == 1:
                                img = ax[i, ..., j, 0]
                                plt.cla()
                                pylab.imshow(img, cmap=pylab.cm.bone)
                                pylab.savefig(out_full_name, bbox_inches='tight')
                            elif shape[4] == 4:
                                img = ax[i, ..., j, 1:4]
                                plt.cla()
                                pylab.imshow(img)
                                pylab.savefig(out_full_name, bbox_inches='tight')
                            elif shape[4] == 3:
                                img = ax[i, ..., j, :]
                                plt.cla()
                                pylab.imshow(img)
                                pylab.savefig(out_full_name, bbox_inches='tight')

            x = ax

        return x

    def random_transform_array(self, x, seed=None):
        """Randomly augment a image array tensor.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_slice_axis = self.slice_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        s, h, w = x.shape[img_slice_axis], x.shape[img_row_axis], x.shape[img_col_axis]
        if transform_matrix is not None:
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            for i in range(s):
                x[...,i,:] = apply_transform(x[...,i,:], transform_matrix, img_channel_axis-1,
                                    fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            for i in range(s):
                x[...,i,:] = random_channel_shift(x[...,i,:],
                                         self.channel_shift_range,
                                         img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                for i in range(s):
                    x[...,i,:] = flip_axis(x[...,i,:], img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                for i in range(s):
                    x[...,i,:] = flip_axis(x[...,i,:], img_row_axis)

        return x

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return ImageArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

class CardiacTimeSeriesDataGenerator(ImageDataGenerator):
    #Customized data augmentation method.

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None,
            toDir=None):
        """Fits internal statistics to some sample data.

        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            x: Numpy array, the data to fit on. Should have rank 5.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        # Raises
            ValueError: in case of invalid input `x`.
        """

        data_format = self.data_format

        if data_format == 'channels_first':
            self.channel_axis = 2
            self.row_axis = 3
            self.col_axis = 4
            self.phase_axis = 1
        if data_format == 'channels_last':
            self.channel_axis = 4
            self.row_axis = 2
            self.col_axis = 3
            self.phase_axis = 1
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 5:
            raise ValueError('Input to `.fit()` should have rank 5. '
                             'Got array with shape: ' + str(x.shape))

        if seed is not None:
            np.random.seed(seed)

        if toDir != None:
            if not os.path.exists(toDir):
                os.makedirs(toDir)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform_array(x[i])
                    if toDir != None:
                        for j in range(x.shape[self.phase_axis]):
                            filename = 'img-%d-%d.png' % (i + r * x.shape[0], j)
                            out_full_name = os.path.join(toDir, filename)
                            shape = ax.shape
                            if shape[self.channel_axis] == 1:
                                img = ax[i, j, ..., 0]
                                plt.cla()
                                pylab.imshow(img, cmap=pylab.cm.bone)
                                pylab.savefig(out_full_name, bbox_inches='tight')
                            elif shape[4] == 4:
                                img = ax[i, j, ..., 1:4]
                                plt.cla()
                                pylab.imshow(img)
                                pylab.savefig(out_full_name, bbox_inches='tight')
                            elif shape[4] == 3:
                                img = ax[i, j, ..., :]
                                plt.cla()
                                pylab.imshow(img)
                                pylab.savefig(out_full_name, bbox_inches='tight')

            x = ax

        return x

    def random_transform_array(self, x, seed=None):
        """Randomly augment a image array tensor.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_phase_axis = self.phase_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        p, h, w = x.shape[img_phase_axis], x.shape[img_row_axis], x.shape[img_col_axis]
        if transform_matrix is not None:
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            for i in range(p):
                x[i,...] = apply_transform(x[i,...], transform_matrix, img_channel_axis-1,
                                    fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            for i in range(p):
                x[i,...] = random_channel_shift(x[i,...],
                                         self.channel_shift_range,
                                         img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                for i in range(p):
                    x[i,...] = flip_axis(x[i,...], img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                for i in range(p):
                    x[i,...] = flip_axis(x[i,...], img_row_axis)

        return x

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return ImageArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)
