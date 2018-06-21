import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from plant.data import cnn_forward
from plant.data import cnn_backward
import os

IMAGE_SIZE = 28


def per_image_standardization(img):
    '''''stat = ImageStat.Stat(img)
    mean = stat.mean
    stddev = stat.stddev
    img = (np.array(img) - stat.mean)/stat.stddev'''
    channel = 3
    num_compare = IMAGE_SIZE * IMAGE_SIZE * channel
    img_arr = np.array(img)
    # img_arr=np.flip(img_arr,2)
    img_t = (img_arr - np.mean(img_arr)) / max(np.std(img_arr), 1 / num_compare)
    return img_t


def restore_model(picPath):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
        y = cnn_forward.forward(x)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(cnn_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(cnn_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                img = cv2.imread(picPath)
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                img_ready = per_image_standardization(img).reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 3))
                preValue = sess.run(preValue, feed_dict={x: img_ready})
                # cv_img = cv2.imread(picPath)
                # cv2.imshow(str(preValue[0]), cv_img)
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def main():
    d = {}
    for index, x in enumerate(os.listdir('train')):
        d[index] = x
    test = pd.read_csv('sample_submission.csv')
    test['species'] = [d[restore_model(os.path.join('test', j))[0]] for j in test['file']]
    test.to_csv('cnn.csv', index=False)


if __name__ == '__main__':
    main()