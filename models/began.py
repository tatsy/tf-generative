import numpy as np
import tensorflow as tf

from .base import BaseModel
from .utils import *

def repelling_regularizer(x, batchsize):
    dims = x.get_shape()[1]
    S_i = tf.tile(x, [batchsize, 1])
    S_j = tf.tile(x, [1, batchsize])
    S_j = tf.reshape(S_j, [-1, dims])
    S_i_T_S_j = tf.reduce_sum(tf.multiply(S_i, S_j), axis=1)
    S_i_norm2 = tf.reduce_sum(tf.square(S_i), axis=1)
    S_j_norm2 = tf.reduce_sum(tf.square(S_j), axis=1)
    f_PT = tf.square(S_i_T_S_j) / (tf.multiply(S_i_norm2, S_j_norm2) + 1.0e-8)
    f_PT = tf.reduce_sum(f_PT) / tf.cast(batchsize * (batchsize - 1), 'float32')
    return f_PT

class Generator(object):
    def __init__(self, input_shape, z_dims):
        self.variables = None
        self.update_ops = None
        self.reuse = False
        self.input_shape = input_shape
        self.z_dims = z_dims
        self.name = 'generator'

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('deconv1'):
                w = self.input_shape[0] // (2 ** 3)
                x = tf.reshape(inputs, [-1, 1, 1, self.z_dims])
                x = tf.layers.conv2d_transpose(x, 512, (w, w), (1, 1), 'valid',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv2'):
                x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv3'):
                x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv4'):
                x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('deconv5'):
                d = self.input_shape[2]
                x = tf.layers.conv2d(x, d, (5, 5), (1, 1), 'same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.tanh(x)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
        self.reuse = True
        return x

class Discriminator(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.variables = None
        self.update_ops = None
        self.reuse = False
        self.name = 'discriminator'

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), 'same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

                S = tf.contrib.layers.flatten(x)

            with tf.variable_scope('deconv1'):
                x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('deconv2'):
                x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('deconv3'):
                x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('deconv4'):
                d = self.input_shape[2]
                x = tf.layers.conv2d_transpose(x, d, (5, 5), (1, 1), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.tanh(x)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
        self.reuse = True
        return x, S

class BEGAN(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='began',
        **kwargs
    ):
        super(BEGAN, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.beta = 0.01
        self.boundary_equil = True
        self.margin = 0.1
        self.update_k_t = None
        self.k_t = tf.Variable(0.5, name='k_t')
        self.lambda_k = 1.0e-4
        self.gamma = 0.7

        self.gen_trainer = None
        self.dis_trainer = None
        self.gen_loss_D = None
        self.gen_loss_G = None
        self.dis_loss = None

        self.f_gen = None
        self.f_dis = None

        self.x_train = None
        self.z_D = None
        self.z_G = None

        self.z_test = None
        self.x_test = None
        self.x_tile = None

        self.train_op = None

        self.build_model()

    def train_on_batch(self, x_batch, index):
        batchsize = x_batch.shape[0]
        z_D = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims))
        z_G = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims))

        # Training
        _, g_loss, _, d_loss = self.sess.run(
            (self.train_op, self.gen_loss_G, self.gen_loss_D, self.dis_loss),
            feed_dict={
                self.x_train: x_batch,
                self.z_G: z_G,
                self.z_D: z_D,
            }
        )

        # Summary update
        if index // 1000 != (index - batchsize) // 1000:
            summary = self.sess.run(
                self.summary,
                feed_dict={
                    self.x_train: x_batch,
                    self.z_D: z_D,
                    self.z_G: z_G,
                    self.z_test: self.test_data
                }
            )
            self.writer.add_summary(summary, index)

        return [
            ('g_loss', g_loss),
            ('d_loss', d_loss)
        ]

    def predict(self, z_samples):
        x_sample = self.sess.run(
            (self.x_test),
            feed_dict={self.z_test: z_samples}
        )
        return x_sample

    def make_test_data(self):
        self.test_data = np.random.uniform(-1, 1, size=(self.test_size * self.test_size, self.z_dims))

    def build_model(self):
        # Trainer
        self.f_dis = Discriminator(self.input_shape)
        self.f_gen = Generator(self.input_shape, self.z_dims)

        x_shape = (self.batchsize,) + self.input_shape
        self.x_train = tf.placeholder(tf.float32, shape=(self.batchsize,) + self.input_shape, name='x_train')

        z_shape = (self.batchsize, self.z_dims)
        self.z_D = tf.placeholder(tf.float32, shape=z_shape, name='z_D')
        self.z_G = tf.placeholder(tf.float32, shape=z_shape, name='z_G')

        x_f_D = self.f_gen(self.z_D)
        x_f_D_pred, _ = self.f_dis(x_f_D)

        x_f_G = self.f_gen(self.z_G)
        x_f_G_pred, S = self.f_dis(x_f_G)

        x_train_pred, _ = self.f_dis(self.x_train)

        f_PT = repelling_regularizer(S, self.batchsize)

        self.gen_loss_D = tf.losses.absolute_difference(x_f_D, x_f_D_pred)
        self.gen_loss_G = tf.losses.absolute_difference(x_f_G, x_f_G_pred)
        self.dis_loss = tf.losses.absolute_difference(self.x_train, x_train_pred)

        gen_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)
        dis_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)

        if self.boundary_equil:
            self.gen_trainer = gen_opt.minimize(self.gen_loss_G + self.beta * f_PT, var_list=self.f_gen.variables)
            self.dis_trainer = dis_opt.minimize(self.dis_loss - self.k_t * self.gen_loss_D, var_list=self.f_dis.variables)
            self.update_k_t = self.k_t.assign(tf.clip_by_value(self.k_t + self.lambda_k * (self.gamma * self.dis_loss - self.gen_loss_D), 0.0, 1.0))

            with tf.control_dependencies([self.gen_trainer, self.dis_trainer, self.update_k_t] + \
                                         self.f_dis.update_ops + self.f_gen.update_ops):
                self.train_op = tf.no_op(name='train')

        else:
            self.gen_trainer = gen_opt.minimize(self.gen_loss_G + self.beta * f_PT, var_list=self.f_gen.variables)
            self.dis_trainer = dis_opt.minimize(self.dis_loss - tf.maximum(0.0, self.margin - self.gen_loss_D), var_list=self.f_dis.variables)

            with tf.control_dependencies([self.gen_trainer, self.dis_trainer] + \
                                          self.f_dis.update_ops + self.f_gen.update_ops):
                self.train_op = tf.no_op(name='train')

        # Predictor
        self.z_test = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        self.x_test = self.f_gen(self.z_test)
        self.x_tile = self.image_tiling(self.x_test, self.test_size, self.test_size)

        tf.summary.image('x_real', image_cast(self.x_train), 10)
        tf.summary.image('x_real_rec', image_cast(x_train_pred), 10)
        tf.summary.image('x_fake', image_cast(x_f_G), 10)
        tf.summary.image('x_fake_rec', image_cast(x_f_G_pred), 10)
        tf.summary.image('x_tile', image_cast(self.x_tile), 1)
        tf.summary.scalar('gen_loss', self.gen_loss_G)
        tf.summary.scalar('dis_loss', self.dis_loss)
        tf.summary.scalar('k_t', self.k_t)

        self.summary = tf.summary.merge_all()
