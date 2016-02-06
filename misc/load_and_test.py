#!/usr/bin/env python3
import argparse

import numpy as np

import functions as f
import layers as l
import network as n
import utils as u


def cnn(weights):
    conv = l.ConvolutionalLayer(2, kernel_size=5, init_func=f.zero, act_func=f.sigmoid)
    fcl = l.FullyConnectedLayer(height=10, init_func=f.zero, act_func=f.softmax)

    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        conv,
        l.MaxPoolingLayer(pool_size=3),
        fcl
    ], f.categorical_crossentropy)

    conv.w = weights["w"][0][0]
    conv.b = np.expand_dims(weights["w"][0][1], 1)
    fcl.w = np.swapaxes(weights["w"][1][0], 0, 1)
    fcl.b = np.expand_dims(weights["w"][1][1], 1)

    return net

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("data", help="the path to the MNIST data set in .npz format (generated using utils.py)")
parser.add_argument("func", help="the function name of the test to be run")
args = parser.parse_args()

trn_set, tst_set = u.load_mnist_npz(args.data)

weights = np.load("weights.npz")
net = locals()[args.func](weights)

print("Testing network...")
accuracy = n.test(net, tst_set)
print("Test accuracy:", accuracy)
