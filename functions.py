import numpy as np


### activations functions #####################################################

def sigmoid(x):
    """Return the sigmoid function"""
    return 1.0 / (1.0 + np.exp(-x))


def der_sigmoid(x):
    """Return the derivative of the sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x, norm_const):
    """"Return the softmax function"""
    return np.exp(x) / norm_const


def der_softmax(x):
    """Return the derivative of the softmax function"""
    return softmax(x, sum(np.exp(x)) * (1 - softmax(x, sum(np.exp(x)))))


def rect_lin(x):
    """Return the rectified linear function"""
    return max(0, x)


def der_rect_lin(x):
    """Return the derivative of the rectified linear function"""
    assert x != 0
    return 1.0 if x > 0 else 0.0


### cost functions ############################################################

def quadratic(a, y):
    """Return the quadratic cost function computed on one observation only"""
    return 0.5 * sum((y - a) ** 2)


def der_quadratic(a, y):
    """Return the derivative of the quadratic cost function"""
    return a - y


def cross_entropy(a, y):
    """Return the cross entropy cost function computed on one observation only"""
    return sum(-y * np.log(a) + (1 - y) * np.log(1 - a))


def der_cross_entropy(a, y):
    """Return the derivative of the cross entropy cost function"""
    return (a - y) / (a - a ** 2)


def log_likelihood(a, y):
    """Return the log likelihood cost function computed on one observation only"""
    return -np.log(a[np.where(y == 1)])


def der_log_likelihood(a, y):
    """Return the derivative of the log likelihood cost function"""
    raise NotImplementedError()


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
