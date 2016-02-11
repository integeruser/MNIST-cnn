#!/usr/bin/env python3
import argparse
import inspect
import sys

import numpy as np

import functions as f
import layers as l
import optimizers as o
import network as n
import utils as u


def fcl01():
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.FullyConnectedLayer(100, init_func=f.glorot_uniform, act_func=f.sigmoid),
        l.FullyConnectedLayer(10, init_func=f.glorot_uniform, act_func=f.sigmoid)
    ], f.quadratic)
    optimizer = o.SGD(3.0)
    num_epochs = 1
    batch_size = 10
    return net, optimizer, num_epochs, batch_size

def fcl02():
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.FullyConnectedLayer(10, init_func=f.glorot_uniform, act_func=f.softmax)
    ], f.categorical_crossentropy)
    optimizer = o.SGD(0.1)
    num_epochs = 1
    batch_size = 10
    return net, optimizer, num_epochs, batch_size


def cnn01():
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.ConvolutionalLayer(2, kernel_size=5, init_func=f.glorot_uniform, act_func=f.sigmoid),
        l.MaxPoolingLayer(pool_size=2),
        l.FullyConnectedLayer(height=10, init_func=f.glorot_uniform, act_func=f.softmax)
    ], f.log_likelihood)
    optimizer = o.SGD(0.1)
    num_epochs = 3
    batch_size = 10
    return net, optimizer, num_epochs, batch_size

def cnn02():
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.ConvolutionalLayer(2, kernel_size=5, init_func=f.glorot_uniform, act_func=f.sigmoid),
        l.MaxPoolingLayer(pool_size=3),
        l.FullyConnectedLayer(height=10, init_func=f.glorot_uniform, act_func=f.softmax)
    ], f.categorical_crossentropy)
    optimizer = o.SGD(0.1)
    num_epochs = 2
    batch_size = 8
    return net, optimizer, num_epochs, batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="the path to the MNIST data set in .npz format (generated using utils.py)")
    parser.add_argument("func", help="the function name of the example to be run")
    args = parser.parse_args()

    np.random.seed(314)

    u.print("Loading '%s'..." % args.data, bcolor=u.bcolors.BOLD)
    trn_set, tst_set = u.load_mnist_npz(args.data)
    trn_set, vld_set = (trn_set[0][:50000], trn_set[1][:50000]), (trn_set[0][50000:], trn_set[1][50000:])

    u.print("Loading '%s'..." % args.func, bcolor=u.bcolors.BOLD)
    net, optimizer, num_epochs, batch_size = locals()[args.func]()
    u.print(inspect.getsource(locals()[args.func]).strip())

    u.print("Training network...", bcolor=u.bcolors.BOLD)
    n.train(net, optimizer, num_epochs, batch_size, trn_set, vld_set)

    u.print("Testing network...", bcolor=u.bcolors.BOLD)
    accuracy = n.test(net, tst_set)
    u.print("Test accuracy: %0.2f%%" % (accuracy*100))
