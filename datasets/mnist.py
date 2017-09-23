import os
import sys
import gzip
import struct
import requests

import numpy as np

import tensorflow as tf

from .datasets import ConditionalDataset
url = 'http://yann.lecun.com/exdb/mnist/'
x_train_file = 'train-images-idx3-ubyte.gz'
y_train_file = 'train-labels-idx1-ubyte.gz'
x_test_file = 't10k-images-idx3-ubyte.gz'
y_test_file = 't10k-labels-idx1-ubyte.gz'

curdir = os.path.abspath(os.path.dirname(__file__))
outdir = os.path.join(curdir, 'files', 'mnist')

CHUNK_SIZE = 32768

def download_mnist():
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Download files
    files = [x_train_file, y_train_file, x_test_file, y_test_file]
    for f in files:
        session = requests.Session()
        response = session.get(os.path.join(url, f), stream=True)
        print('Downloading: %s' % (os.path.join(url, f)))
        with open(os.path.join(outdir, f), 'wb') as fp:
            dl = 0
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    dl += len(chunk)
                    fp.write(chunk)

                    mb = dl / 1.0e6
                    sys.stdout.write('\r%.2f MB downloaded...' % (mb))
                    sys.stdout.flush()

            sys.stdout.write('\nFinish!\n')
            sys.stdout.flush()

def load_images(filename):
    with gzip.GzipFile(filename, 'rb') as fp:
        # Magic number
        magic = struct.unpack('>I', fp.read(4))[0]

        # item sizes
        n, rows, cols = struct.unpack('>III', fp.read(4 * 3))

        # Load items
        data = np.ndarray((n, rows, cols), dtype=np.uint8)
        for i in range(n):
            sub = struct.unpack('B' * rows * cols, fp.read(rows * cols))
            data[i] = np.asarray(sub).reshape((rows, cols))

        return data

def load_labels(filename):
    with gzip.GzipFile(filename, 'rb') as fp:
        # Magic number
        magic = struct.unpack('>I', fp.read(4))

        # item sizes
        n= struct.unpack('>I', fp.read(4))[0]

        # Load items
        data = np.zeros((n, 10), dtype=np.uint8)
        for i in range(n):
            b = struct.unpack('>B', fp.read(1))[0]
            data[i, b] = 1

        return data

def load_data():
    if not os.path.exists(outdir):
        download_mnist()

    x_train = load_images(os.path.join(outdir, x_train_file))
    y_train = load_labels(os.path.join(outdir, y_train_file))

    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
    x_train = (x_train[:, :, :, np.newaxis] / 255.0).astype('float32')
    y_train = y_train.astype('float32')

    datasets = ConditionalDataset()
    datasets.images = x_train
    datasets.attrs = y_train
    datasets.attr_names = [str(i) for i in range(10)]

    return datasets
