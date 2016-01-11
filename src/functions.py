import numpy as np


### weights initializations ####################################################

def glorot_uniform(shape, num_neurons_in, num_neurons_out):
    scale = np.sqrt(6. / (num_neurons_in + num_neurons_out))
    return np.random.uniform(low=-scale, high=scale, size=shape)


def zero(shape, *args):
    return np.zeros(shape)


### activations ################################################################

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def der_sigmoid(x, y=None):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def der_softmax(x, y=None):
    s = softmax(x)
    if y is not None:
        k = s[np.where(y == 1)]
        a = - k * s
        a[np.where(y == 1)] = k * (1 - k)
        return a
    return s * (1 - s)


### objectives #################################################################

def quadratic(a, y):
    return a-y

def log_likelihood(a, y):
    return -1.0 / a[np.where(y == 1)]

def categorical_crossentropy(a, y):
    a = a.flatten() / np.sum(a)
    i = np.where(y.flatten() == 1)
    return np.log(a)[i]
