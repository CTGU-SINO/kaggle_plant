import tensorflow as tf
import re

KERNEL_SIZE = 5
CON2D_LAYER1 = 32
CON2D_LAYER2 = 64
FULLY_NODE = 1024
OUTPUT_SIZE = 12
TOWER_NAME = 'tower'
INITIALIZER_CON2D = tf.contrib.layers.xavier_initializer_conv2d()
INITIALIZER_FULLY = tf.contrib.layers.xavier_initializer()


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


# def _variable_on_cpu(name, shape, initializer):
#     with tf.device('/cpu:0'):
#         dtype = tf.float32
#         var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
#     return var
#
#
# def _variable_with_weight_decay(name, shape, initializer, wd):
#     var = _variable_on_cpu(name, shape, initializer)
#     if wd is not None:
#         weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
#         tf.add_to_collection('losses', weight_decay)
#     return var


def conv_op(x, name, n_out, training, useBN, kh=5, kw=5, dh=1, dw=1, padding="SAME", activation=tf.nn.relu):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                            initializer=INITIALIZER_CON2D)
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))
        con2d = tf.nn.conv2d(x, w, [1, dh, dw, 1], padding=padding)
        z = tf.nn.bias_add(con2d, b)
        if useBN:
            z = tf.layers.batch_normalization(z, trainable=training)
        if activation:
            z = activation(z)

        _activation_summary(z)
    return z


def max_pool_op(x, name, kh=2, kw=2, dh=2, dw=2, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding=padding,
                          name=name)


def fc_op(x, name, n_out):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        w = tf.get_variable(scope + "w", shape=[n_in, n_out],
                            dtype=tf.float32,
                            initializer=INITIALIZER_FULLY)
        b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.01))

        fc = tf.matmul(x, w) + b

        _activation_summary(fc)

    return fc


def bn(x, beta_name, gamma_name, name="bn"):
    axes = [d for d in range(len(x.get_shape()))]
    beta = tf.get_variable(beta_name, shape=[], initializer=tf.constant_initializer(0.0))
    gamma = tf.get_variable(gamma_name, shape=[], initializer=tf.constant_initializer(1.0))
    x_mean, x_variance = tf.nn.moments(x, axes)
    y = tf.nn.batch_normalization(x, x_mean, x_variance, beta, gamma, 1e-10, name)
    return y


def forward(x):
    con2d1 = conv_op(x, "Con2d_layer1", CON2D_LAYER1, True, True, KERNEL_SIZE, KERNEL_SIZE, 1, 1)
    max_pool1 = max_pool_op(con2d1, 'max_pooling1')
    con2d2 = conv_op(max_pool1, "Con2d_layer2", CON2D_LAYER2, True, True, KERNEL_SIZE, KERNEL_SIZE, 1, 1)
    max_pool2 = max_pool_op(con2d2, 'max_pooling2')
    flat = tf.reshape(max_pool2, [-1, 7 * 7 * CON2D_LAYER2])
    fully1 = fc_op(flat, 'Fully_layer1', FULLY_NODE)
    prediction = fc_op(fully1, 'SoftMax', OUTPUT_SIZE)
    return prediction