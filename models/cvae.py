import numpy as np
import tensorflow as tf

from .base import CondBaseModel
from .utils import *

class Encoder(object):
    def __init__(self, input_shape, z_dims, num_attrs):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.z_dims = z_dims
        self.num_attrs = num_attrs

    def __call__(self, inputs, attrs, training=True):
        with tf.variable_scope('encoder', reuse=self.reuse):
            with tf.variable_scope('conv1'):
                a = tf.reshape(attrs, [-1, 1, 1, self.num_attrs])
                a = tf.tile(a, [1, self.input_shape[0], self.input_shape[1], 1])
                x = tf.concat([inputs, a], axis=-1)
                x = tf.layers.conv2d(x, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('fc1'):
                x = tf.contrib.layers.flatten(x)
                x = tf.layers.dense(x, 1024)
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('fc2'):
                z_avg = tf.layers.dense(x, self.z_dims)
                z_log_var = tf.layers.dense(x, self.z_dims)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        self.reuse = True

        return z_avg, z_log_var

class Decoder(object):
    def __init__(self, input_shape):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape

    def __call__(self, inputs, attrs, training=True):
        with tf.variable_scope('decoder', reuse=self.reuse):
            with tf.variable_scope('fc1'):
                w = self.input_shape[0] // (2 ** 3)
                x = tf.concat([inputs, attrs], axis=-1)
                x = tf.layers.dense(x, w * w * 256)
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)
                x = tf.reshape(x, [-1, w, w, 256])

            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv4'):
                d = self.input_shape[2]
                x = tf.layers.conv2d_transpose(x, d, (3, 3), (1, 1), 'same')
                x = tf.tanh(x)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
        self.reuse = True

        return x

class CVAE(CondBaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='cvae',
        **kwargs
    ):
        super(CVAE, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.total_loss = None
        self.optimizer = None

        self.encoder = None
        self.decoder = None

        self.x_train = None
        self.c_train = None

        self.z_test = None
        self.x_test = None
        self.c_test = None

        self.build_model()

    def train_on_batch(self, batch, index):
        x_batch, c_batch = batch

        _, loss, summary = self.sess.run(
            (self.optimizer, self.total_loss, self.summary),
            feed_dict={self.x_train: x_batch, self.c_train: c_batch, self.z_test: self.test_data['z_test'], self.c_test: self.test_data['c_test']}
        )

        self.writer.add_summary(summary, index)
        return [ ('loss', loss) ]

    def predict(self, batch):
        z_samples, c_samples = batch
        x_sample = self.sess.run(
            self.x_test,
            feed_dict={self.z_test: z_samples, self.c_test: c_samples}
        )
        return x_sample

    def make_test_data(self):
        c_t = np.identity(self.num_attrs)
        c_t = np.tile(c_t, (self.test_size, 1))
        z_t = np.random.normal(size=(self.test_size, self.z_dims))
        z_t = np.tile(z_t, (1, self.num_attrs))
        z_t = z_t.reshape((self.test_size * self.num_attrs, -1))
        self.test_data = {'z_test': z_t, 'c_test': c_t}

    def build_model(self):
        self.encoder = Encoder(self.input_shape, self.z_dims, self.num_attrs)
        self.decoder = Decoder(self.input_shape)

        # Trainer
        batch_shape = (None,) + self.input_shape
        self.x_train = tf.placeholder(tf.float32, shape=batch_shape)
        self.c_train = tf.placeholder(tf.float32, shape=(None, self.num_attrs))

        z_avg, z_log_var = self.encoder(self.x_train, self.c_train)
        epsilon = tf.random_normal(tf.shape(z_avg))
        z_sample = z_avg + tf.multiply(tf.exp(0.5 * z_log_var), epsilon)
        x_sample = self.decoder(z_sample, self.c_train)

        self.total_loss = tf.constant(0.0)
        self.total_loss += tf.reduce_mean(tf.squared_difference(self.x_train, x_sample))
        self.total_loss += kl_loss(z_avg, z_log_var)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=2.0e-4, beta1=0.5).minimize(self.total_loss)

        # Predictor
        self.z_test = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        self.c_test = tf.placeholder(tf.float32, shape=(None, self.num_attrs))

        self.x_test = self.decoder(self.z_test, self.c_test)
        x_tile = self.image_tiling(self.x_test)

        # Summary
        tf.summary.image('x_real', self.x_train, 10)
        tf.summary.image('x_fake', x_sample, 10)
        tf.summary.image('x_tile', x_tile, 1)
        tf.summary.scalar('total_loss', self.total_loss)

        self.summary = tf.summary.merge_all()