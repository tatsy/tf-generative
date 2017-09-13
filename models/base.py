import os
import sys
import time
import math
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

from abc import ABCMeta, abstractmethod
from .utils import *

class BaseModel(metaclass=ABCMeta):
    """
    Base class for non-conditional generative networks
    """

    def __init__(self, **kwargs):
        """
        Initialization
        """
        if 'name' not in kwargs:
            raise Exception('Please specify model name!')

        self.name = kwargs['name']

        if 'input_shape' not in kwargs:
            raise Exception('Please specify input shape!')

        self.check_input_shape(kwargs['input_shape'])
        self.input_shape = kwargs['input_shape']

        if 'output' not in kwargs:
            self.output = 'output'
        else:
            self.output = kwargs['output']

        self.resume = kwargs['resume']

        self.sess = tf.Session()
        self.writer = None
        self.summary = None

        self.test_size = 100
        self.test_data = None

        self.test_mode = False

    def check_input_shape(self, input_shape):
        # Check for CelebA
        if input_shape == (64, 64, 3):
            return

        # Check for MNIST (size modified)
        if input_shape == (32, 32, 1):
            return

        # Check for Cifar10, 100 etc
        if input_shape == (32, 32, 3):
            return

        errmsg = 'Input size should be 32 x 32 or 64 x 64!'
        raise Exception(errmsg)

    def main_loop(self, datasets, samples, epochs=100, batchsize=50):
        """
        Main learning loop
        """
        # Create output directories if not exist
        out_dir = os.path.join(self.output, self.name)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        res_out_dir = os.path.join(out_dir, 'results')
        if not os.path.isdir(res_out_dir):
            os.makedirs(res_out_dir)

        chk_out_dir = os.path.join(out_dir, 'checkpoints')
        if not os.path.isdir(chk_out_dir):
            os.makedirs(chk_out_dir)

        time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_out_dir = os.path.join(out_dir, 'log', time_str)
        if not os.path.isdir(log_out_dir):
            os.makedirs(log_out_dir)

        # Make test data
        self.make_test_data()

        # Start training
        with self.sess.as_default():
            current_epoch = tf.Variable(0, name='current_epoch', dtype=tf.int32)
            current_batch = tf.Variable(0, name='current_batch', dtype=tf.int32)

            # Initialize global variables
            if self.resume is not None:
                print('Resume training: %s' % self.resume)
                self.load_model(self.resume)
            else:
                self.sess.run(tf.global_variables_initializer())

            self.writer = tf.summary.FileWriter(log_out_dir, self.sess.graph)

            print('\n\n--- START TRAINING ---\n')
            num_data = len(datasets)
            for e in range(current_epoch.eval(), epochs):
                self.sess.run(current_epoch.assign(e))

                perm = np.random.permutation(num_data)
                start_time = time.time()
                for b in range(current_batch.eval(), num_data, batchsize):
                    self.sess.run(current_batch.assign(b))

                    bsize = min(batchsize, num_data - b)
                    indx = perm[b:b+bsize]

                    # Get batch and train on it
                    x_batch = self.make_batch(datasets, indx)
                    losses = self.train_on_batch(x_batch, e * num_data + (b + bsize))

                    # Print current status
                    elapsed_time = time.time() - start_time
                    eta = elapsed_time / (b + bsize) * (num_data - (b + bsize))
                    ratio = 100.0 * (b + bsize) / num_data
                    print(chr(27) + "[2K", end='')
                    print('Epoch #%d,  Batch: %d / %d (%6.2f %%)  ETA: %s' % \
                          (e + 1, b + bsize, num_data, ratio, time_format(eta)))

                    for i, (k, v) in enumerate(losses):
                        text = '%s = %8.6f' % (k, v)
                        print('  %25s' % (text), end='')
                        if (i + 1) % 3 == 0:
                            print('')

                    print('\n')
                    sys.stdout.flush()

                    # Save generated images
                    if (b + bsize) % 10000 == 0 or (b + bsize) == num_data:
                        outfile = os.path.join(res_out_dir, 'epoch_%04d_batch_%d.png' % (e + 1, b + bsize))
                        self.save_images(samples, outfile)
                        outfile = os.path.join(chk_out_dir, 'epoch_%04d' % (e + 1))
                        self.save_model(outfile)

                    if self.test_mode:
                        print('\nFinish testing: %s' % self.name)
                        return

                print('')
                self.sess.run(current_batch.assign(0))

    def make_batch(self, datasets, indx):
        """
        Get batch from datasets
        """
        return datasets[indx]

    def save_images(self, samples, filename):
        """
        Save images generated from random sample numbers
        """
        imgs = self.predict(samples) * 0.5 + 0.5
        imgs = np.clip(imgs, 0.0, 1.0)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))

        fig = plt.figure(figsize=(8, 8))
        grid = gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)
        for i in range(100):
            ax = plt.Subplot(fig, grid[i])
            if imgs.ndim == 4:
                ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            else:
                ax.imshow(imgs[i, :, :], cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        fig.savefig(filename, dpi=200)
        plt.close(fig)

    def save_model(self, model_file):
        saver = tf.train.Saver()
        saver.save(self.sess, model_file)

    def load_model(self, model_file):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_file)

    @abstractmethod
    def make_test_data(self):
        """
        Please override "make_test_data" method in the derived model!
        """
        pass

    @abstractmethod
    def predict(self, z_sample):
        """
        Please override "predict" method in the derived model!
        """
        pass

    @abstractmethod
    def train_on_batch(self, x_batch, index):
        """
        Please override "train_on_batch" method in the derived model!
        """
        pass

    def image_tiling(self, images):
        n_images = self.test_size
        img_arr = tf.split(images, n_images, 0)

        size = int(math.ceil(math.sqrt(n_images)))
        while len(img_arr) < size * size:
            img_arr.append(tf.zeros(self.input_shape, tf.float32))

        rows = []
        for i in range(size):
            rows.append(tf.concat(img_arr[i * size : (i + 1) * size], axis=1))

        tile = tf.concat(rows, axis=2)
        return tile

class CondBaseModel(BaseModel):
    def __init__(self, **kwargs):
        super(CondBaseModel, self).__init__(**kwargs)

        if 'attr_names' not in kwargs:
            raise Exception('Please specify attribute names (attr_names')
        self.attr_names = kwargs['attr_names']
        self.num_attrs = len(self.attr_names)

        self.test_size = 10

    def make_batch(self, datasets, indx):
        images = datasets.images[indx]
        attrs = datasets.attrs[indx]
        return images, attrs

    def save_images(self, samples, filename):
        assert self.attr_names is not None

        num_samples = len(samples)
        attrs = np.identity(self.num_attrs)
        attrs = np.tile(attrs, (num_samples, 1))

        samples = np.tile(samples, (1, self.num_attrs))
        samples = samples.reshape((num_samples * self.num_attrs, -1))

        imgs = self.predict([samples, attrs]) * 0.5 + 0.5
        imgs = np.clip(imgs, 0.0, 1.0)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))

        fig = plt.figure(figsize=(self.num_attrs, 10))
        grid = gridspec.GridSpec(num_samples, self.num_attrs, wspace=0.1, hspace=0.1)
        for i in range(num_samples * self.num_attrs):
            ax = plt.Subplot(fig, grid[i])
            if imgs.ndim == 4:
                ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            else:
                ax.imshow(imgs[i, :, :], cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        fig.savefig(filename, dpi=200)
        plt.close(fig)

    def image_tiling(self, images):
        n_images = self.test_size * self.num_attrs
        img_arr = tf.split(images, n_images, 0)

        rows = []
        for i in range(self.test_size):
            rows.append(tf.concat(img_arr[i * self.num_attrs: (i + 1) * self.num_attrs], axis=2))

        tile = tf.concat(rows, axis=1)
        return tile