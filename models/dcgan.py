import numpy as np
import tensorflow as tf

from .base import BaseModel
from .utils import *
from .wnorm import *

class Generator(object):
    def __init__(self, input_shape, z_dims, use_wnorm=False):
        self.variables = None
        self.update_ops = None
        self.reuse = False
        self.use_wnorm = use_wnorm
        self.input_shape = input_shape
        self.z_dims = z_dims

    def __call__(self, inputs, training=True):
        with tf.variable_scope('generator', reuse=self.reuse):
            with tf.variable_scope('fc1'):
                w = self.input_shape[0] // (2 ** 3)
                x = tf.reshape(inputs, [-1, 1, 1, self.z_dims])
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, 256, (w, w), (1, 1),
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                else:
                    x = tf.layers.conv2d(x, 256, (w, w), (1, 1),
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.batch_normalization(x, training=training)

                x = tf.nn.relu(x)

            with tf.variable_scope('conv1'):
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, 256, (5, 5), (2, 2), 'same', use_scale=False,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                else:
                    x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same', kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv2'):
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, 128, (5, 5), (2, 2), 'same', use_scale=False,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                else:
                    x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same', kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv3'):
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, 64, (5, 5), (2, 2), 'same', use_scale=False,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                else:
                    x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same', kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv4'):
                d = self.input_shape[2]
                if self.use_wnorm:
                    x = conv2d_transpose_wnorm(x, d, (5, 5), (1, 1), 'same', use_scale=True,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                else:
                    x = tf.layers.conv2d_transpose(x, d, (5, 5), (1, 1), 'same',
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.tanh(x)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        self.reuse = True
        return x

class Discriminator(object):
    def __init__(self, input_shape, use_wnorm=False):
        self.input_shape = input_shape
        self.variables = None
        self.update_ops = None
        self.use_wnorm = use_wnorm
        self.reuse = False

    def __call__(self, inputs, training=True):
        with tf.variable_scope('discriminator', reuse=self.reuse):
            with tf.variable_scope('conv1'):
                if self.use_wnorm:
                    x = conv2d_wnorm(inputs, 64, (5, 5), (2, 2), 'same', use_scale=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                else:
                    x = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), 'same', kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv2'):
                if self.use_wnorm:
                    x = conv2d_wnorm(x, 128, (5, 5), (2, 2), 'same', use_scale=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                else:
                    x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same', kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                if self.use_wnorm:
                    x = conv2d_wnorm(x, 256, (5, 5), (2, 2), 'same', use_scale=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                else:
                    x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same', kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv4'):
                if self.use_wnorm:
                    x = conv2d_wnorm(x, 512, (5, 5), (2, 2), 'same', use_scale=False,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                else:
                    x = tf.layers.conv2d(x, 512, (5, 5), (2, 2), 'same', kernel_initializer=tf.contrib.layers.xavier_initializer())
                    x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv5'):
                w = self.input_shape[0] // (2 ** 4)
                if self.use_wnorm:
                    y = conv2d_wnorm(x, 1, (w, w), (1, 1), 'valid', use_scale=True,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                else:
                    y = tf.layers.conv2d(x, 1, (w, w), (1, 1), 'valid',
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
                y = tf.reshape(y, [-1, 1])

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
        self.reuse = True
        return y

class DCGAN(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='dcgan',
        **kwargs
    ):
        super(DCGAN, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims
        self.use_wnorm = True

        self.gen_loss = None
        self.dis_loss = None
        self.gen_optimizer = None
        self.dis_optimizer = None
        self.train_op = None

        self.gen_acc = None
        self.dis_acc = None

        self.x_train = None

        self.z_test = None
        self.x_test = None

        self.build_model()

    def train_on_batch(self, x_batch, index):
        batchsize = x_batch.shape[0]
        z_sample = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims))

        _, g_loss, d_loss, g_acc, d_acc, summary = self.sess.run(
            (self.train_op, self.gen_loss, self.dis_loss, self.gen_acc, self.dis_acc, self.summary),
            feed_dict={self.x_train: x_batch, self.z_train: z_sample, self.z_test: self.test_data}
        )

        self.writer.add_summary(summary, index)

        return [
            ('g_loss', g_loss), ('d_loss', d_loss),
            ('g_acc',  g_acc), ('d_acc', d_acc)
        ]

    def predict(self, z_samples):
        x_sample = self.sess.run(
            self.x_test,
            feed_dict={self.z_test: z_samples}
        )
        return x_sample

    def make_test_data(self):
        self.test_data = np.random.uniform(-1.0, 1.0, size=(self.test_size * self.test_size, self.z_dims))

    def build_model(self):
        # Trainer
        self.f_dis = Discriminator(self.input_shape, use_wnorm=self.use_wnorm)
        self.f_gen = Generator(self.input_shape, self.z_dims, use_wnorm=self.use_wnorm)

        batch_shape = (None, self.z_dims)
        self.z_train = tf.placeholder(tf.float32, shape=batch_shape)
        x_fake = self.f_gen(self.z_train)
        y_fake = self.f_dis(x_fake)

        self.gen_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_fake), y_fake)
        self.gen_optimizer = tf.train.AdamOptimizer(2.0e-4, beta1=0.5) \
                             .minimize(self.gen_loss, var_list=self.f_gen.variables)

        batch_shape = (None,) + self.input_shape
        self.x_train = tf.placeholder(tf.float32, batch_shape)

        y_real = self.f_dis(self.x_train)
        self.dis_loss = 0.5 * tf.losses.sigmoid_cross_entropy(tf.ones_like(y_real), y_real) + \
                        0.5 * tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_fake), y_fake)
        self.dis_optimizer = tf.train.AdamOptimizer(2.0e-4, beta1=0.5) \
                             .minimize(self.dis_loss, var_list=self.f_dis.variables)

        self.gen_acc = binary_accuracy(tf.ones_like(y_fake), y_fake)
        self.dis_acc = 0.5 * binary_accuracy(tf.ones_like(y_real), y_real) + \
                       0.5 * binary_accuracy(tf.zeros_like(y_fake), y_fake)

        with tf.control_dependencies([self.gen_optimizer, self.dis_optimizer] + \
                                      self.f_dis.update_ops + \
                                      self.f_gen.update_ops):
            self.train_op = tf.no_op(name='train')

        # Predictor
        self.z_test = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        self.x_test = self.f_gen(self.z_test, training=False)

        x_tile = self.image_tiling(self.x_test, self.test_size, self.test_size)

        tf.summary.image('x_real', self.x_train, 10)
        tf.summary.image('x_fake', image_cast(x_fake), 10)
        tf.summary.image('x_tile', image_cast(x_tile), 1)
        tf.summary.scalar('gen_loss', self.gen_loss)
        tf.summary.scalar('dis_loss', self.dis_loss)
        tf.summary.scalar('gen_acc', self.gen_acc)
        tf.summary.scalar('dis_acc', self.dis_acc)
        self.summary = tf.summary.merge_all()
