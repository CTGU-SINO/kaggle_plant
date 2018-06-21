import os
import cv2
import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 制作TFRecord格式
def createTFRecord(type):
    cwd = os.getcwd()
    file_path = os.path.join(cwd, type)
    writer = tf.python_io.TFRecordWriter(type + '.TFRecords')
    for index, name in enumerate(os.listdir(file_path)):
        jpg_path = os.path.join(file_path, name)
        for img_name in os.listdir(jpg_path):
            print(img_name)
            img_path = os.path.join(jpg_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (28, 28))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(index),
                'img_raw': _bytes_feature(img_raw)
            }))
            writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename, num_epochs, shuffle):
    reader = tf.TFRecordReader()
    print('num_epochs', num_epochs)
    filename_queue = tf.train.string_input_producer([filename], shuffle=shuffle, num_epochs=num_epochs)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'img_raw': tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28, 3])

    img = tf.image.per_image_standardization(img)
    labels = tf.cast(features['label'], tf.int32)
    return img, labels


def createBatch(filename, batch_size, shuffle, num_epochs=None):
    images, labels = read_and_decode(filename, num_epochs, shuffle)

    min_train_examples = 5000
    min_test_examples = 5000

    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                          batch_size=batch_size,
                                                          capacity=min_train_examples + 3 * batch_size,
                                                          min_after_dequeue=min_train_examples
                                                          )
    else:
        image_batch, label_batch = tf.train.batch(
            [images, labels],
            batch_size=batch_size,
            capacity=min_test_examples + 3 * batch_size)

    label_batch = tf.one_hot(label_batch, depth=12, dtype=tf.int32)
    tf.summary.image('images', image_batch)
    return image_batch, label_batch


def test_batch():
    TRAIN_DIR = 'train.TFRecords'
    BATCH_SIZE = 50
    flag = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        images, labels = createBatch(TRAIN_DIR, BATCH_SIZE, True)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while flag:
                flag = False
                print(labels)
                label = sess.run([labels])
                image = sess.run([images])
                print((image[0] * 255).astype(np.uint8))
                img = cv2.imshow('x', (image[0] * 255).astype(np.uint8))
                print(label[0])
        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
        coord.join(threads)