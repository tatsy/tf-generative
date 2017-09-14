import numpy as np
import tensorflow as tf

from .base import BaseModel
from .utils import *

class Generator(object):
    def __init__(self, input_shape):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.name = 'generator'

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('fc1'):
                w = self.input_shape[0] // (2 ** 3)
                x = tf.layers.dense(inputs, w * w * 256)
                x = tf.nn.relu(x)
                x = tf.reshape(x, [-1, w, w, 256])

            with tf.variable_scope('deconv1'):
                x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = lrelu(x)

            with tf.variable_scope('deconv2'):
                x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = lrelu(x)

            with tf.variable_scope('deconv3'):
                x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = lrelu(x)

            with tf.variable_scope('deconv4'):
                d = self.input_shape[2]
                x = tf.layers.conv2d_transpose(x, d, (3, 3), (1, 1), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.tanh(x)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True
        return x

class Discriminator(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.variables = None
        self.reuse = False
        self.name = 'discriminator'

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), 'same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = lrelu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = lrelu(x)

            with tf.variable_scope('fc1'):
                x = tf.contrib.layers.flatten(x)
                x = tf.layers.dense(x, 1024, kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = lrelu(x)

            with tf.variable_scope('fc2'):
                w = self.input_shape[0] // (2 ** 3)
                x = tf.layers.dense(x, w * w * 256, kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = lrelu(x)
                x = tf.reshape(x, (-1, w, w, 256))

            with tf.variable_scope('deconv1'):
                x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = lrelu(x)

            with tf.variable_scope('deconv2'):
                x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = lrelu(x)

            with tf.variable_scope('deconv3'):
                x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = lrelu(x)

            with tf.variable_scope('deconv4'):
                d = self.input_shape[2]
                x = tf.layers.conv2d_transpose(x, d, (5, 5), (1, 1), 'same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.tanh(x)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.reuse = True
        return x

class BEGAN(BaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='began',
        **kwargs
    ):
        super(BEGAN, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.boundary_equil = True
        self.margin = 0.1
        self.update_k_t = None
        self.k_t = tf.Variable(0.0, name='k_t')
        self.lambda_k = 1.0e-3
        self.gamma = 0.5

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

        self.build_model()

    def train_on_batch(self, x_batch, index):
        batchsize = x_batch.shape[0]
        z_D = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims))
        z_G = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims))

        if index // 1000 != (index + batchsize) // 1000:
            # Training w/ summary
            _, _, g_loss_D, g_loss_G, d_loss, _, summary = self.sess.run(
                (self.gen_trainer, self.dis_trainer, self.gen_loss_D, self.gen_loss_D, self.dis_loss, self.update_k_t, self.summary),
                feed_dict={
                    self.x_train: x_batch,
                    self.z_D: z_D,
                    self.z_G: z_G,
                    self.z_test: self.test_data
                }
            )
            self.writer.add_summary(summary, index)
        else:
            # Training w/o summary
            _, _, g_loss_D, g_loss_G, d_loss, _ = self.sess.run(
                (self.gen_trainer, self.dis_trainer, self.gen_loss_D, self.gen_loss_D, self.dis_loss, self.update_k_t),
                feed_dict={
                    self.x_train: x_batch,
                    self.z_D: z_D,
                    self.z_G: z_G
                }
            )

        return [
            ('g_loss', g_loss_G),
            ('d_loss', d_loss)
        ]

    def predict(self, z_samples):
        x_sample = self.sess.run(
            (self.x_test),
            feed_dict={self.z_test: z_samples}
        )
        return x_sample

    def make_test_data(self):
        self.test_data = np.random.uniform(-1, 1, size=(self.test_size, self.z_dims))

    def build_model(self):
        # Trainer
        self.f_dis = Discriminator(self.input_shape)
        self.f_gen = Generator(self.input_shape)

        self.z_D = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        self.z_G = tf.placeholder(tf.float32, shape=(None, self.z_dims))

        x_f_D = self.f_gen(self.z_D)
        x_f_D_pred = self.f_dis(x_f_D)

        x_f_G = self.f_gen(self.z_G)
        x_f_G_pred = self.f_dis(x_f_G)

        self.x_train = tf.placeholder(tf.float32, shape=(None,) + self.input_shape)
        x_train_pred = self.f_dis(self.x_train)

        self.gen_loss_D = tf.losses.absolute_difference(x_f_D, x_f_D_pred)
        self.gen_loss_G = tf.losses.absolute_difference(x_f_G, x_f_G_pred)
        self.dis_loss = tf.losses.absolute_difference(self.x_train, x_train_pred)

        gen_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)
        dis_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)

        if self.boundary_equil:
            self.gen_trainer = gen_opt.minimize(self.gen_loss_G, var_list=self.f_gen.variables)
            self.dis_trainer = dis_opt.minimize(self.dis_loss - self.k_t * self.gen_loss_D, var_list=self.f_dis.variables)
            self.update_k_t = self.k_t.assign(tf.clip_by_value(self.k_t + self.lambda_k * (self.gamma * self.dis_loss - self.gen_loss_G), 0.0, 1.0))
        else:
            self.gen_trainer = gen_opt.minimize(self.gen_loss_G, var_list=self.f_gen.variables)
            self.dis_trainer = dis_opt.minimize(self.dis_loss - tf.maximum(0.0, self.margin - self.gen_loss_D), var_list=self.f_dis.variables)

        # Predictor
        self.z_test = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        self.x_test = self.f_gen(self.z_test)
        self.x_tile = self.image_tiling(self.x_test)

        tf.summary.image('x_real', self.x_train, 10)
        tf.summary.image('x_fake', x_f_G, 10)
        tf.summary.image('x_tile', self.x_tile, 1)
        tf.summary.scalar('gen_loss', self.gen_loss_G)
        tf.summary.scalar('dis_loss', self.dis_loss)
        tf.summary.scalar('k_t', self.k_t)

        self.summary = tf.summary.merge_all()
