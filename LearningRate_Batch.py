from keras.callbacks import *

class LearningRateBatchScheduler(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """
    current_epoch = 0
    current_iter = 0

    def __init__(self, schedule):
        super(LearningRateBatchScheduler, self).__init__()
        self.schedule = schedule

    def on_batch_end(self, curr_iter, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.schedule(self.current_epoch, curr_iter)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)


    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        self.current_epoch = epoch