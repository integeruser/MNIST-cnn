__author__ = 'F. Cagnin and A. Torcinovich'

import numpy as np


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
