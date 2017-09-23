import os
import sys
import time
import math
import numpy as np
from PIL import Image

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
        self.saver = None
        self.summary = None

        self.test_size = 10
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

    def main_loop(self, datasets, epochs=100, batchsize=50):
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
            self.saver = tf.train.Saver()
            if self.resume is not None:
                print('Resume training: %s' % self.resume)
                self.load_model(self.resume)
            else:
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(tf.local_variables_initializer())

            # Update rule
            num_data = len(datasets)
            update_epoch = current_epoch.assign(current_epoch + 1)
            update_batch = current_batch.assign(tf.mod(tf.minimum(current_batch + batchsize, num_data), num_data))

            self.writer = tf.summary.FileWriter(log_out_dir, self.sess.graph)
            self.sess.graph.finalize()

            print('\n\n--- START TRAINING ---\n')
            for e in range(current_epoch.eval(), epochs):
                perm = np.random.permutation(num_data)
                start_time = time.time()
                for b in range(current_batch.eval(), num_data, batchsize):
                    bsize = min(batchsize, num_data - b)
                    indx = perm[b:b+bsize]

                    # Get batch and train on it
                    x_batch = self.make_batch(datasets, indx)
                    losses = self.train_on_batch(x_batch, e * num_data + (b + bsize))

                    # Print current status
                    elapsed_time = time.time() - start_time
                    eta = elapsed_time / (b + bsize) * (num_data - (b + bsize))
                    ratio = 100.0 * (b + bsize) / num_data
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
                    save_period = 10000
                    if b != 0 and ((b // save_period != (b + bsize) // save_period) or ((b + bsize) == num_data)):
                        outfile = os.path.join(res_out_dir, 'epoch_%04d_batch_%d.png' % (e + 1, b + bsize))
                        self.save_images(outfile)
                        outfile = os.path.join(chk_out_dir, 'epoch_%04d' % (e + 1))
                        self.save_model(outfile)

                    if self.test_mode:
                        print('\nFinish testing: %s' % self.name)
                        return

                    # Update batch index
                    self.sess.run(update_batch)

                print('')
                self.sess.run(update_epoch)

    def make_batch(self, datasets, indx):
        """
        Get batch from datasets
        """
        return datasets.images[indx]

    def save_images(self, filename):
        """
        Save images generated from random sample numbers
        """
        imgs = self.predict(self.test_data) * 0.5 + 0.5
        imgs = np.clip(imgs, 0.0, 1.0)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))

        _, height, width, dims = imgs.shape

        margin = min(width, height) // 10
        figure = np.ones(((margin + height) * 10 + margin, (margin + width) * 10 + margin, dims), np.float32)

        for i in range(100):
            row = i // 10
            col = i % 10

            y = margin + (margin + height) * row
            x = margin + (margin + width) * col
            figure[y:y+height, x:x+width, :] = imgs[i, :, :, :]

        figure = Image.fromarray((figure * 255.0).astype(np.uint8))
        figure.save(filename)

    def save_model(self, model_file):
        self.saver.save(self.sess, model_file)

    def load_model(self, model_file):
        self.saver.restore(self.sess, model_file)

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

    def image_tiling(self, images, rows, cols):
        n_images = rows * cols
        mg = max(self.input_shape[0], self.input_shape[1]) // 20
        pad_img = tf.pad(images, [[0, 0], [mg, mg], [mg, mg], [0, 0]], constant_values=1.0)
        img_arr = tf.split(pad_img, n_images, 0)

        rows = []
        for i in range(self.test_size):
            rows.append(tf.concat(img_arr[i * cols: (i + 1) * cols], axis=2))

        tile = tf.concat(rows, axis=1)
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

    def save_images(self, filename):
        assert self.attr_names is not None

        try:
            samples = self.test_data['z_test']
        except KeyError as e:
            print('The key "z_test" is not found in test_data')

        try:
            attrs = self.test_data['c_test']
        except KeyError as e:
            print('The key "c_test" is not found in test_data')

        imgs = self.predict([samples, attrs]) * 0.5 + 0.5
        imgs = np.clip(imgs, 0.0, 1.0)

        _, height, width, dims = imgs.shape

        margin = min(width, height) // 10
        figure = np.ones(((margin + height) * self.test_size + margin, (margin + width) * self.num_attrs + margin, dims), np.float32)

        for i in range(self.test_size * self.num_attrs):
            row = i // self.num_attrs
            col = i % self.num_attrs

            y = margin + (margin + height) * row
            x = margin + (margin + width) * col
            figure[y:y+height, x:x+width, :] = imgs[i, :, :, :]

        figure = Image.fromarray((figure * 255.0).astype(np.uint8))
        figure.save(filename)
