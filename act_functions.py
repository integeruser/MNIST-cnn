__author__ = 'F. Cagnin and A. Torcinovich'

import numpy as np


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
    if (x == 0.0):
        raise ValueError("derivative of this value does not exist")
    return 1.0 if x > 0 else 0.0
