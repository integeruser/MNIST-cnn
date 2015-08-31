__author__ = 'F. Cagnin and A. Torcinovich'

import numpy as np


def sigmoid(x):
    """Return the sigmoid function"""
    return 1.0 / (1.0 + np.exp(-x))


vec_sigmoid = np.vectorize(
    sigmoid, doc="Return the vectorized version of the sigmoid function")


def der_sigmoid(x):
    """Return the derivative of the sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))


vec_der_sigmoid = np.vectorize(
    der_sigmoid, doc="Return the vectorized version of the  derivative of the sigmoid function")


def softmax(x, norm_const):
    """"Return the softmax function"""
    return np.exp(x) / norm_const


vec_softmax = np.vectorize(
    softmax, doc="Return the vectorized version of the softmax function")


def der_softmax(x):
    """Return the derivative of the softmax function"""
    return softmax(x, sum(np.exp(x)) * (1 - vec_softmax(x, sum(np.exp(x)))))


vec_der_softmax = np.vectorize(
    der_softmax, doc="Return the vectorized version of the derivative of the softmax function")


def rect_lin(x):
    """Return the rectified linear function"""
    return max(0, x)


vec_rect_lin = np.vectorize(
    rect_lin, doc="Return the vectorized version of the rectified linear function")


def der_rect_lin(x):
    """Return the derivative of the rectified linear function"""
    if (x == 0.0):
        raise ValueError("derivative of this value does not exist")
    return 1.0 if x > 0 else 0.0


vec_der_rect_lin = np.vectorize(
    der_rect_lin, doc="Return the vectorized version of the derivative of the rectified linear function"
)
