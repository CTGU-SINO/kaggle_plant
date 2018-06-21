import tensorflow as tf
from plant.data.tfread import createBatch
from plant.data import cnn_forward
import os

IMAGE_SIZE = 28
BATCH_SIZE = 50
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
STEPS = 1000000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./cnn_model/"
CHECK_POINT = './cnn'
MODEL_NAME = "mnist_model"
TRAIN_DIR = 'train.TFRecords'


def backward():
    graph = tf.Graph()
    tf.reset_default_graph()
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            with tf.device('/cpu:0'):
                images, labels = createBatch(TRAIN_DIR, BATCH_SIZE, True)
            y = cnn_forward.forward(images)
            global_step = tf.Variable(0, trainable=False)

            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels, 1), logits=y)
            # loss = tf.reduce_mean(ce, name="loss")
            cem = tf.reduce_mean(ce)
            loss = cem + tf.reduce_sum(tf.get_collection('losses'))

            learning_rate = tf.train.exponential_decay(
                LEARNING_RATE_BASE,
                global_step,
                5000 / BATCH_SIZE,
                LEARNING_RATE_DECAY,
                staircase=True)

            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

            ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            ema_op = ema.apply(tf.trainable_variables())
            with tf.control_dependencies([train_step, ema_op]):
                train_op = tf.no_op(name='train')

            saver = tf.train.Saver()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
            if not os.path.exists(CHECK_POINT):
                os.mkdir(CHECK_POINT)

            tf.summary.merge_all()
            tf.summary.FileWriter(CHECK_POINT + '/summary', graph)
            try:
                for i in range(STEPS):
                    _, loss_value, step = sess.run([train_op, loss, global_step])
                    if i % 500 == 0:
                        print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            except tf.errors.OutOfRangeError:
                print("done!")
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    backward()