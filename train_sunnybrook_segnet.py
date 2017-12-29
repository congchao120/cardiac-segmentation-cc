import os

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tfmodel
import numpy as np

DATA_NAME = 'Data'
TRAIN_SOURCE = "Training"
TEST_SOURCE = 'Testing'
ONLINE_SOURCE = 'Online'
RUN_NAME = "SELU_Run03"
OUTPUT_NAME = 'Output'
CHECKPOINT_FN = 'model.ckpt'

WORKING_DIR = os.getcwd()

TRAIN_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TRAIN_SOURCE)
TEST_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TEST_SOURCE)
ONLINE_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, ONLINE_SOURCE)

ROOT_LOG_DIR = os.path.join(WORKING_DIR, OUTPUT_NAME)
LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

TRAIN_WRITER_DIR = os.path.join(LOG_DIR, TRAIN_SOURCE)
TEST_WRITER_DIR = os.path.join(LOG_DIR, TEST_SOURCE)

NUM_EPOCHS = 10
MAX_STEP = 5000
BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
LEARNING_RATE = 1e-04

SAVE_RESULTS_INTERVAL = 5
SAVE_CHECKPOINT_INTERVAL = 100

CROP_SIZE = 128


def main():
    train_data = tfmodel.GetData(TRAIN_DATA_DIR)
    test_data = tfmodel.GetData(TEST_DATA_DIR)
    online_data = tfmodel.GetData(ONLINE_DATA_DIR)
    if not os.path.exists(ROOT_LOG_DIR):
        os.makedirs(ROOT_LOG_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(TRAIN_WRITER_DIR):
        os.makedirs(TRAIN_WRITER_DIR)
    if not os.path.exists(TEST_WRITER_DIR):
        os.makedirs(TEST_WRITER_DIR)

    g = tf.Graph()

    with g.as_default():

        images, labels = tfmodel.placeholder_inputs(batch_size=BATCH_SIZE)

        logits, softmax_logits = tfmodel.inference(images, class_inc_bg=2, crop_size=CROP_SIZE)

        tfmodel.add_output_images(images=images, logits=softmax_logits, labels=labels)

        loss = tfmodel.loss_dice(logits=softmax_logits, labels=labels, crop_size=CROP_SIZE)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_op = tfmodel.training(loss=loss, learning_rate=1e-04, global_step=global_step)

        accuracy = tfmodel.eval_dice(logits=softmax_logits, labels=labels, crop_size=CROP_SIZE, smooth=1.0)

        accuracy_array = tfmodel.eval_dice_array(logits=softmax_logits, labels=labels, crop_size=CROP_SIZE, smooth=1.0)

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.global_variables())

    sm = tf.train.SessionManager(graph=g)

    with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR) as sess:

        sess.run(tf.local_variables_initializer())

        train_writer = tf.summary.FileWriter(TRAIN_WRITER_DIR, sess.graph)
        test_writer = tf.summary.FileWriter(TEST_WRITER_DIR)

        global_step_value, = sess.run([global_step])

        print("Last trained iteration was: ", global_step_value)

        #try:

        while True:

            if global_step_value >= MAX_STEP:
                print(f"Reached MAX_STEP: {MAX_STEP} at step: {global_step_value}")
                break

            images_batch, labels_batch, _ = train_data.next_batch(BATCH_SIZE)
            feed_dict = {images: images_batch, labels: labels_batch}

            if (global_step_value + 1) % SAVE_RESULTS_INTERVAL == 0:
                _, loss_value, accuracy_value, global_step_value, summary_str = sess.run(
                    [train_op, loss, accuracy, global_step, summary], feed_dict=feed_dict)
                train_writer.add_summary(summary_str, global_step=global_step_value)
                print(f"TRAIN Step: {global_step_value}\tLoss: {loss_value}\tAccuracy: {accuracy_value}")

                images_batch, labels_batch, _  = test_data.next_batch(TEST_BATCH_SIZE)
                feed_dict = {images: images_batch, labels: labels_batch}

                loss_value, accuracy_value, global_step_value, summary_str = sess.run(
                    [loss, accuracy, global_step, summary], feed_dict=feed_dict)
                test_writer.add_summary(summary_str, global_step=global_step_value)
                print(f"TEST  Step: {global_step_value}\tLoss: {loss_value}\tAccuracy: {accuracy_value}")

            else:
                _, loss_value, accuracy_value, global_step_value = sess.run([train_op, loss, accuracy, global_step],
                                                                            feed_dict=feed_dict)
                print(f"TRAIN Step: {global_step_value}\tLoss: {loss_value}\tAccuracy: {accuracy_value}")

            if global_step_value % SAVE_CHECKPOINT_INTERVAL == 0:
                saver.save(sess, CHECKPOINT_FL, global_step=global_step_value)
                print("Checkpoint Saved")

        evalArr = []
        fileArr = []
        for index in range(int(online_data.examples/TEST_BATCH_SIZE -5)):
            images_batch, labels_batch, files_batch = online_data.next_batch(TEST_BATCH_SIZE)
            feed_dict = {images: images_batch, labels: labels_batch}
            logits, loss_value, accuracy = sess.run(
                [softmax_logits, loss, accuracy_array], feed_dict=feed_dict)

            tfmodel.save_output_images(images=images_batch, logits=logits, image_names=files_batch,
                                       contour_type='i')

            evalArr = np.append(evalArr, list(accuracy))
            fileArr = np.append(fileArr, list(files_batch))

        save_dir = 'D:\cardiac_data\Sunnybrook\Sunnybrook_online_submission'
        detail_eval = os.path.join(save_dir, 'evaluation_detail_{:s}.csv'.format('i'))
        resArr = np.transpose([fileArr, evalArr])
        np.savetxt(detail_eval, resArr, fmt='%s', delimiter=',')
        #except Exception as e:
        #    print('Exception')
        #    print(e)



        train_writer.flush()
        test_writer.flush()
        saver.save(sess, CHECKPOINT_FL, global_step=global_step_value)
        print("Checkpoint Saved")

        print("Stopping")


if __name__ == '__main__':
    main()