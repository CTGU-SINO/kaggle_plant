import time
import tensorflow as tf
from plant.data.tfread import createBatch
from plant.data import cnn_forward
from plant.data import cnn_backward

TEST_INTERVAL_SECS = 20
IMAGE_SIZE = 28
BATCH_SIZE = 120
EVAL_DIR = 'eval.TFRecords'


def test():
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            with tf.device('/cpu:0'):
                images, labels = createBatch(EVAL_DIR, BATCH_SIZE, True)

            y = cnn_forward.forward(images)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            while True:
                try:
                    ema = tf.train.ExponentialMovingAverage(cnn_backward.MOVING_AVERAGE_DECAY)
                    ema_restore = ema.variables_to_restore()
                    saver = tf.train.Saver(ema_restore)

                    ckpt = tf.train.get_checkpoint_state(cnn_backward.MODEL_SAVE_PATH)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                        accuracy_score = sess.run(accuracy)
                        print("After {} training step(s), test accuracy = {}".format(global_step, accuracy_score))
                    else:
                        print('No checkpoint file found')
                        return

                    time.sleep(TEST_INTERVAL_SECS)
                except tf.errors.OutOfRangeError:
                    print("done!")
                finally:
                    coord.request_stop()
                coord.join(threads)


if __name__ == '__main__':
    test()