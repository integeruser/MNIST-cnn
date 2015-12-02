import numpy as np


def get_derivative(func):
    derivatives = {
        sigmoid: der_sigmoid,
        softmax: der_softmax,

        quadratic: der_quadratic,
        cross_entropy: der_cross_entropy,
        log_likelihood: der_log_likelihood,

        max: der_max
    }
    assert func in derivatives
    return derivatives[func]


### activations functions #####################################################

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def der_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)

def der_softmax(x):
    s = softmax(x)
    return s * (1 - s)


### cost functions (computed on a single observation) #########################

def quadratic(a, y):
    return 0.5 * sum((y - a) ** 2)

def der_quadratic(a, y):
    return a - y


def cross_entropy(a, y):
    return sum(-y * np.log(a) + (1 - y) * np.log(1 - a))

def der_cross_entropy(a, y):
    return (a - y) / (a - a ** 2)


def log_likelihood(a, y):
    return -np.log(a[np.where(y == 1)])

def der_log_likelihood(a, y):
    raise NotImplementedError


### polling functions #########################################################

def max(x):
    return np.max(x)

def der_max(x):
    der = np.zeros_like(x, dtype=np.uint8)
    der[np.unravel_index(x.argmax(), x.shape)] = 1
    return der
