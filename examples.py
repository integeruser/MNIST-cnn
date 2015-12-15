#!/usr/bin/env python3
import argparse
import inspect
import sys

import numpy as np

import functions as f
import layers as l
import network as n
import utils as u


def fcl01():  # 91.75%
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.FullyConnectedLayer(height=10, act_func="softmax")
    ], "log_likelihood")
    optimizer = {"type": "SGD", "eta": 0.1}
    num_epochs = 1
    batch_size = 10
    return net, optimizer, num_epochs, batch_size

def cnn01():  # 88.13%
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.ConvolutionalLayer(depth=2, kernel_size=5, act_func="sigmoid"),
        l.PollingLayer(window_size=2, poll_func="max"),
        l.FullyConnectedLayer(height=10, act_func="softmax")
    ], "log_likelihood")
    optimizer = {"type": "SGD", "eta": 0.1}
    num_epochs = 1
    batch_size = 10
    return net, optimizer, num_epochs, batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="the path to the MNIST data set in .npz format (generated using utils.py)")
    parser.add_argument("func", help="the function name of the example to be run")
    args = parser.parse_args()

    np.random.seed(314)

    u.print("Loading '%s'..." % args.data, bcolor=u.bcolors.BOLD)
    trn_set, tst_set = u.load_mnist_npz(args.data)

    u.print("Loading '%s'..." % args.func, bcolor=u.bcolors.BOLD)
    net, optimizer, num_epochs, batch_size = locals()[args.func]()
    u.print(inspect.getsource(locals()[args.func]).strip())

    u.print("Training NN...", bcolor=u.bcolors.BOLD)
    n.train(net, optimizer, num_epochs, batch_size, trn_set, vld_set=tst_set)
