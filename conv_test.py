"""
To know more or get code samples, please visit my website:
https://morvanzhou.github.io/tutorials/
Or search: 莫烦Python
Thank you for supporting!
"""

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 8 - RNN LSTM Regressor example

# to try tensorflow, un-comment following two lines
# import os
# os.environ['KERAS_BACKEND']='tensorflow'
import numpy as np
from keras.utils.vis_utils import plot_model
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.layers import Conv1D
BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.mean(seq, axis=(1))
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    return [seq[:, :, np.newaxis], res[:, np.newaxis, np.newaxis], xs]

model = Sequential()
# build a LSTM RNN

model.add(Conv1D(INPUT_SIZE,
                batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
                 kernel_size=TIME_STEPS, padding='same'))
model.add(Conv1D(OUTPUT_SIZE,
                 kernel_size=TIME_STEPS, padding='valid'))
# add output layer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='mean_absolute_error',)

plot_model(model, show_shapes=True, to_file='conv_test.png')
model.summary()
print('Training ------------')
for step in range(2000):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch, Y_batch, xs = get_batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)
    plt.plot(xs[0, :], X_batch[0].flatten(), 'r')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)
    print('Ground truth: {}, predict: {}'.format(Y_batch[10], pred[10]))
    if step % 10 == 0:
        print('train cost: ', cost)