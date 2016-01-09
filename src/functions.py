import numpy as np


### activations functions #####################################################

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


### loss functions (computed on a single observation) #########################

def quadratic(a, y):
    return a - y


def categorical_crossentropy(a, y):
    a_rescaled = np.copy(a) / np.sum(a, axis=-1)
    c = -np.sum(y * np.log(a_rescaled), axis=a_rescaled.ndim - 1)
    return np.mean(c, axis=-1)


def cross_entropy(a, y):
    return (a - y) / (a - a ** 2)


def log_likelihood(a, y):
    return -1.0 / a[np.where(y == 1)]


### pooling functions #########################################################

def max(x):
    return np.max(x)

def der_max(x):
    der = np.zeros_like(x, dtype=np.uint8)
    der[np.unravel_index(x.argmax(), x.shape)] = 1
    return der
