from __future__ import division

import re

import tensorflow as tf

output_size = 64
gf_dim = 64
df_dim = 64
batch_size = 64
c_dim = 3
TOWER_NAME = 'tower'


class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = _variable_on_cpu("beta", [shape[-1]], tf.float32,
                                             initializer=tf.constant_initializer(0.))
                self.gamma = _variable_on_cpu("gamma", [shape[-1]], tf.float32,
                                              initializer=tf.random_normal_initializer(1., 0.02))

                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')

                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
            x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed


d_bn0 = batch_norm(name="d_bn0")
d_bn1 = batch_norm(name="d_bn1")
d_bn2 = batch_norm(name="d_bn2")
d_bn3 = batch_norm(name="d_bn3")

g_bn0 = batch_norm(name="g_bn0")
g_bn1 = batch_norm(name="g_bn1")
g_bn2 = batch_norm(name="g_bn2")
g_bn3 = batch_norm(name="g_bn3")


def _variable_on_cpu(name, shape, dtype, initializer):
    """ if variable has been initialized, return it. Or else, initialize the variable.
    go to tf.get_variable for details.
    """
    with tf.device("/job:ps/task:0"):
        var = tf.get_variable(name, shape, dtype, initializer=initializer)
    return var


def _activation_summary(x):
    """ Output a Summary protocol buffer for output from per layer of the network.
    """
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def generator(z, y=None):
    s = output_size
    s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

    # project `z` and reshape
    z_, h0_w, h0_b = linear(z, gf_dim * 8 * s16 * s16, 'g_h0_lin', with_w=True)

    h0 = tf.reshape(z_, [-1, s16, s16, gf_dim * 8])
    h0 = tf.nn.relu(g_bn0(h0), name="g_h0_relu")
    _activation_summary(h0)
    h1, h1_w, h1_b = deconv2d(h0, [batch_size, s8, s8, gf_dim * 4], name='g_h1',
                              with_w=True)
    h1 = tf.nn.relu(g_bn1(h1),  name="g_h1_relu")

    _activation_summary(h1)

    h2, h2_w, h2_b = deconv2d(h1, [batch_size, s4, s4, gf_dim * 2], name='g_h2',
                              with_w=True)
    h2 = tf.nn.relu(g_bn2(h2),  name="g_h2_relu")
    _activation_summary(h2)
    h3, h3_w, h3_b = deconv2d(h2, [batch_size, s2, s2, gf_dim * 1], name='g_h3',
                              with_w=True)
    h3 = tf.nn.relu(g_bn3(h3),  name="g_h3_relu")
    _activation_summary(h3)

    # h4.shape = [s, s, c_dim]
    h4, h4_w, h4_b = deconv2d(h3, [batch_size, s, s, c_dim], name='g_h4', with_w=True)
    _activation_summary(h4)
    return tf.nn.tanh(h4)


def discriminator(image, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'), name="d_h0_conv")
    h1 = lrelu(d_bn1(conv2d(h0, df_dim * 2, name='d_h1_conv')), name="d_h1_conv")
    h2 = lrelu(d_bn2(conv2d(h1, df_dim * 4, name='d_h2_conv')), name="d_h2_conv")
    h3 = lrelu(d_bn3(conv2d(h2, df_dim * 8, name='d_h3_conv')), name="d_h3_conv")
    h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')
    _activation_summary(h0)
    _activation_summary(h1)
    _activation_summary(h2)
    _activation_summary(h3)
    _activation_summary(h4)
    return tf.nn.sigmoid(h4), h4


def sampler(z):
    tf.get_variable_scope().reuse_variables()

    s = output_size
    s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)

    # project `z` and reshape
    h0 = tf.reshape(linear(z, gf_dim * 8 * s16 * s16, 'g_h0_lin'),
                    [-1, s16, s16, gf_dim * 8])
    h0 = tf.nn.relu(g_bn0(h0, train=False))

    h1 = deconv2d(h0, [batch_size, s8, s8, gf_dim * 4], name='g_h1')
    h1 = tf.nn.relu(g_bn1(h1, train=False))

    h2 = deconv2d(h1, [batch_size, s4, s4, gf_dim * 2], name='g_h2')
    h2 = tf.nn.relu(g_bn2(h2, train=False))

    h3 = deconv2d(h2, [batch_size, s2, s2, gf_dim * 1], name='g_h3')
    h3 = tf.nn.relu(g_bn3(h3, train=False))

    h4 = deconv2d(h3, [batch_size, s, s, c_dim], name='g_h4')

    return tf.nn.tanh(h4)


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear") as scope:

        matrix = _variable_on_cpu("Matrix", [shape[1], output_size], tf.float32,
                                  tf.random_normal_initializer(stddev=stddev))
        bias = _variable_on_cpu("bias", [output_size], tf.float32,
                                initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name) as scope:
        w = _variable_on_cpu('w', [k_h, k_w, input_.get_shape()[-1], output_dim], tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=stddev))
        biases = _variable_on_cpu('biases', [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape(), name=scope.name)

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name)as scope:
        # filter : [height, width, output_channels, in_channels]
        w = _variable_on_cpu('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], tf.float32,
                             initializer=tf.random_normal_initializer(stddev=stddev))
        biases = _variable_on_cpu('biases', [output_shape[-1]], tf.float32, initializer=tf.constant_initializer(0.0))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape(), name=scope.name)

        if with_w:
            return deconv, w, biases
        else:
            return deconv
