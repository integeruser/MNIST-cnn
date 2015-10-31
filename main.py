#!/usr/bin/env python3
import argparse

from layers import *
import network


def load_mnist_dataset(dataset_path):
    import gzip
    import pickle

    def to_categorical(y):
        y = np.asarray(y, dtype="int32")
        nb_classes = np.max(y) + 1
        Y = np.zeros((len(y), nb_classes))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
        return Y

    with gzip.open(dataset_path, "rb") as f:
        tr_d, va_d, te_d = pickle.load(f, encoding="latin1")
        training_inputs = [np.reshape(x, (x.size, 1)) for x in tr_d[0]]
        training_data = zip(training_inputs, (np.reshape(y, (y.size, 1)) for y in to_categorical(tr_d[1])))
        validation_inputs = [np.reshape(x, (x.size, 1)) for x in va_d[0]]
        validation_data = zip(validation_inputs, va_d[1])
        test_inputs = [np.reshape(x, (x.size, 1)) for x in te_d[0]]
        test_data = zip(test_inputs, (np.reshape(y, (y.size, 1)) for y in to_categorical(te_d[1])))
        return training_data, validation_data, test_data


parser = argparse.ArgumentParser()
parser.add_argument("dataset_path")
args = parser.parse_args()

np.random.seed(314)

print("Loading data")
training_data, validation_data, test_data = load_mnist_dataset(args.dataset_path)

print("Generating desired CNN")
net = network.NeuralNetwork([
    InputLayer(height=28, width=28),
    FullyConnectedLayer(height=100, width=1, act_func=functions.sigmoid),
    FullyConnectedLayer(height=10, width=1, act_func=functions.sigmoid)
], functions.quadratic)

print("Training CNN")
network.train(net, list(training_data), 30, 10, 3.0)

print("Testing CNN performances")
network.test(net, list(test_data))
