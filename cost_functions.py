__author__ = 'F. Cagnin and A. Torcinovich'

import numpy as np


# the derivative result must be arrays

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
