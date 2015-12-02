import numpy as np


def get_derivative(func):
    derivatives = {
        sigmoid: der_sigmoid,
        softmax: der_softmax,
        rect_lin: der_rect_lin,

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
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def der_softmax(x):
    return softmax(x) * (1 - softmax(x))


def rect_lin(x):
    return max(0, x)

def der_rect_lin(x):
    assert x != 0
    return 1.0 if x > 0 else 0.0


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
