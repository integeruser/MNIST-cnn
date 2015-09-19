import numpy as np


def get_derivative(func):
    derivatives = {
        sigmoid: der_sigmoid,
        softmax: der_softmax,
        rect_lin: der_rect_lin,

        quadratic: der_quadratic,
        cross_entropy: der_cross_entropy,
        log_likelihood: der_log_likelihood,

        max: der_max,
        mean: der_mean,
        lp_norm: der_lp_norm
    }
    assert func in derivatives
    return derivatives[func]


### activations functions #####################################################

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x, norm_const):
    return np.exp(x) / norm_const


def der_softmax(x):
    return softmax(x, sum(np.exp(x)) * (1 - softmax(x, sum(np.exp(x)))))


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

def max(x, *args):
    return np.max(x)


# TODO: needs the elements of the previous layer
def der_max(x, *args):
    res = np.zeros(x.shape)
    res[x.index(max(x))] = 1
    return res


def mean(x, *args):
    return np.mean(x)


def der_mean(x, *args):
    return np.ones(x.shape) / x.size


# TODO change p with args or something similar
def lp_norm(x, *args):
    return np.linalg.norm(x, args[0])


# TODO needs the elements of the previous layer
def der_lp_norm(x, *args):
    # args stands for p
    return x ** (args[0] - 1) * sum(sum(abs(x) ** args[0])) ** (1 / (args[0] - 1))
