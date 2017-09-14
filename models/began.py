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
                x = tf.layers.dense(inputs, w * w * 256, kernel_initializer=tf.contrib.layers.xavier_initializer())
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)
                x = tf.reshape(x, [-1, w, w, 256])

            with tf.variable_scope('deconv1'):
                x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('deconv2'):
                x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('deconv3'):
                x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('deconv4'):
                d = self.input_shape[2]
                x = tf.layers.conv2d_transpose(x, d, (3, 3), (1, 1), 'same')
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
                x = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('deconv1'):
                x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('deconv2'):
                x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('deconv3'):
                x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('deconv4'):
                d = self.input_shape[2]
                x = tf.layers.conv2d_transpose(x, d, (5, 5), (1, 1), 'same')
                x = tf.layers.batch_normalization(x, training=training)
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

        self.boundary_equiv = True
        self.margin = 0.1
        self.k_t = tf.Variable(0.0, name='k_t')
        self.lambda_k = tf.Variable(1.0e-3, name='lambda_k')
        self.gamma = tf.constant(0.5, name='gamma')

        self.gen_trainer = None
        self.dis_trainer = None
        self.gen_loss = None
        self.dis_loss = None

        self.f_gen = None
        self.f_dis = None

        self.x_train = None

        self.z_test = None
        self.x_test = None

        self.build_model()

    def train_on_batch(self, x_batch, index):
        batchsize = x_batch.shape[0]
        z_sample = np.random.uniform(-1.0, 1.0, size=(batchsize, self.z_dims))

        _, _, g_loss, d_loss, summary = self.sess.run(
            (self.gen_trainer, self.dis_trainer, self.gen_loss, self.dis_loss, self.summary),
            feed_dict={
                self.x_train: x_batch,
                self.z_train: z_sample,
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
            self.x_test,
            feed_dict={self.z_test: z_samples}
        )
        return x_sample

    def make_test_data(self):
        self.test_data = np.random.uniform(-1, 1, size=(self.test_size, self.z_dims))

    def build_model(self):
        # Trainer
        self.f_dis = Discriminator(self.input_shape)
        self.f_gen = Generator(self.input_shape)

        self.z_train = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        x_fake = self.f_gen(self.z_train)
        x_fake_pred = self.f_dis(x_fake)

        self.x_train = tf.placeholder(tf.float32, shape=(None,) + self.input_shape)
        x_train_pred = self.f_dis(self.x_train)

        self.gen_loss = tf.losses.absolute_difference(x_fake, x_fake_pred)
        self.dis_loss = tf.losses.absolute_difference(self.x_train, x_train_pred)

        gen_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)
        dis_opt = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5)

        if self.boundary_equiv:
            self.gen_trainer = gen_opt.minimize(self.gen_loss, var_list=self.f_gen.variables)
            self.dis_trainer = dis_opt.minimize(self.dis_loss - self.k_t * self.gen_loss, var_list=self.f_dis.variables)

            self.k_t = tf.clip_by_value(self.k_t + self.lambda_k * (self.gamma * self.dis_loss - self.gen_loss), 0.0, 1.0)
            tf.summary.scalar('k_t', self.k_t)
        else:
            self.gen_trainer = gen_opt.minimize(self.gen_loss, var_list=self.f_gen.variables)
            self.dis_trainer = dis_opt.minimize(self.dis_loss - tf.maximum(0.0, self.margin - self.gen_loss), var_list=self.f_dis.variables)

        # Predictor
        self.z_test = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        self.x_test = self.f_gen(self.z_test)

        x_tile = self.image_tiling(self.x_test)

        tf.summary.image('x_real', self.x_train, 10)
        tf.summary.image('x_fake', x_fake, 10)
        tf.summary.image('x_tile', x_tile, 1)
        tf.summary.scalar('gen_loss', self.gen_loss)
        tf.summary.scalar('dis_loss', self.dis_loss)

        self.summary = tf.summary.merge_all()
