#!/usr/bin/env python3
import sys

import functions as f
import layers as l
import network as n
import numpy as np
import utils as u

np.random.seed(314)

if len(sys.argv) != 2:
    print("Usage: %s mnist_npzpath")
    sys.exit(2)

mnist_npzpath = sys.argv[1]

print("Loading '%s'..." % mnist_npzpath)
trn_set, tst_set = u.load_mnist_npz(mnist_npzpath)

print("Building CNN...")
net = n.NeuralNetwork([
    l.InputLayer(height=28, width=28),
    l.ConvolutionalLayer(depth=2, height=24, width=24, kernel_size=5, act_func=f.sigmoid),
    l.PollingLayer(depth=2, height=12, width=12, window_size=2, poll_func=f.max),
    l.FullyConnectedLayer(height=100, width=1, act_func=f.sigmoid),
    l.FullyConnectedLayer(height=10, width=1, act_func=f.sigmoid)
], f.quadratic)
print(net)

print("Training CNN...")
n.train(net, trn_set, 1, 10, 0.1)

print("Testing CNN...")
n.test(net, tst_set)
