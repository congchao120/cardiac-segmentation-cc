import tensorflow as tf

def loss_calc(logits, labels):

    class_inc_bg = 2

    labels = labels[...,0]

    class_weights = tf.constant([[10.0/90, 10.0]])

    onehot_labels = tf.one_hot(labels, class_inc_bg)

    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)

    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)

    weighted_losses = unweighted_losses * weights

    loss = tf.reduce_mean(weighted_losses)

    tf.summary.scalar('loss', loss)
    return loss


def evaluation(logits, labels):
    labels = labels[..., 0]

    correct_prediction = tf.equal(tf.argmax(logits, 3), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def eval_dice(logits, labels, crop_size, smooth):

    labels = tf.image.resize_image_with_crop_or_pad(labels, crop_size, crop_size)

    axes = (1, 2)
    y_true = tf.cast(labels[..., 0], tf.float32)
    y_pred = tf.cast(logits[..., 1], tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    summation = tf.reduce_sum(y_true * y_true, axis=axes) + tf.reduce_sum(y_pred * y_pred, axis=axes)

    dice = tf.reduce_mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)

    return dice

def eval_dice_array(logits, labels, crop_size, smooth):

    labels = tf.image.resize_image_with_crop_or_pad(labels, crop_size, crop_size)

    axes = (1, 2)
    y_true = tf.cast(labels[..., 0], tf.float32)
    y_pred = tf.cast(logits[..., 1], tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    summation = tf.reduce_sum(y_true * y_true, axis=axes) + tf.reduce_sum(y_pred * y_pred, axis=axes)

    dice = (2.0 * intersection + smooth) / (summation + smooth)

    return dice

def loss_dice(logits, labels, crop_size):
    return 1.0 - eval_dice(logits=logits, labels=labels, crop_size=crop_size, smooth=1.0)