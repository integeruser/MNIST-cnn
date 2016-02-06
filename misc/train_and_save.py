#!/usr/bin/env python3
import argparse

import numpy as np
np.random.seed(1337)

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD

import utils as u


def fcl(trn_set, tst_set):
    trn_x, trn_y = trn_set
    trn_x = np.squeeze(trn_x, axis=1).reshape((60000, 784))
    trn_y = np.squeeze(trn_y, axis=2)
    tst_x, tst_y = tst_set
    tst_x = np.squeeze(tst_x, axis=1).reshape((10000, 784))
    tst_y = np.squeeze(tst_y, axis=2)

    model = Sequential()

    model.add(Dense(1024, activation="sigmoid", init="zero", input_dim=28*28))
    model.add(Dense(10,   activation="softmax", init="zero"))

    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.1))
    return model, trn_x, trn_y, tst_x, tst_y


def cnn(trn_set, tst_set):
    trn_x, trn_y = trn_set
    trn_y = np.squeeze(trn_y, axis=2)
    tst_x, tst_y = tst_set
    tst_y = np.squeeze(tst_y, axis=2)

    model = Sequential()

    model.add(Convolution2D(2, 5, 5, activation='sigmoid', input_shape=(1, 28, 28)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))
    return model, trn_x, trn_y, tst_x, tst_y

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("data", help="the path to the MNIST data set in .npz format (generated using utils.py)")
parser.add_argument("func", help="the function name of the test to be run")
args = parser.parse_args()

trn_set, tst_set = u.load_mnist_npz(args.data)
model, trn_x, trn_y, tst_x, tst_y = locals()[args.func](trn_set, tst_set)
model.fit(trn_x, trn_y, nb_epoch=1, batch_size=10, show_accuracy=True)
score = model.evaluate(tst_x, tst_y, show_accuracy=True)
print("Test score:", score[0])
print("Test accuracy:", score[1])

# save weights for comparison
with open("weights.npz", "wb") as f:
    w = list()
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            print(layer.get_config())
            w.append(weights)
    np.savez_compressed(f, w=w)
