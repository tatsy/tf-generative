import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf

from models import *
from datasets import load_data, mnist, svhn

models = {
    'vae': VAE,
    'dcgan': DCGAN,
    'began': BEGAN,
    'wgan': WGAN,
    'cvae': CVAE,
    'cvaegan': CVAEGAN
}

def main(_):
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training GANs or VAEs')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--output', default='output')
    parser.add_argument('--zdims', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--testmode', action='store_true')

    args = parser.parse_args()

    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # Load datasets
    if args.dataset == 'mnist':
        datasets = mnist.load_data()
    elif args.dataset == 'svhn':
        datasets = svhn.load_data()
    else:
        datasets = load_data(args.dataset)

    # Construct model
    if args.model not in models:
        raise Exception('Unknown model:', args.model)

    model = models[args.model](
        input_shape=datasets.shape[1:],
        attr_names=datasets.attr_names or None,
        z_dims=args.zdims,
        output=args.output,
        resume=args.resume
    )

    if args.testmode:
        model.test_mode = True

    tf.set_random_seed(12345)

    # Training loop
    datasets.images = datasets.images * 2.0 - 1.0
    model.main_loop(datasets,
                    epochs=args.epoch,
                    batchsize=args.batchsize)

if __name__ == '__main__':
    tf.app.run(main)