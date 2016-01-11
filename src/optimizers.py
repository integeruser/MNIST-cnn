import abc

import numpy as np


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def apply(self, layers, der_weights, der_biases, batch_len):
        raise AssertionError


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def apply(self, layers, der_weights, der_biases, batch_len):
        for _, layer in layers:
            gw = der_weights[layer]/batch_len
            layer.w += -(self.lr*gw)

            gb = der_biases[layer] /batch_len
            layer.b += -(self.lr*gb)


class Adadelta(Optimizer):
    def __init__(self):
        self.rho = 0.95
        self.eps = 1e-8

    def apply(self, layers, der_weights, der_biases, batch_len):
        gsum_weights = {layer: 0 for _, layer in layers}
        xsum_weights = {layer: 0 for _, layer in layers}
        gsum_biases  = {layer: 0 for _, layer in layers}
        xsum_biases  = {layer: 0 for _, layer in layers}

        for _, layer in layers:
            gw = der_weights[layer]/batch_len
            gsum_weights[layer] = rho*gsum_weights[layer] + (1-rho)*gw*gw
            dx = -np.sqrt((xsum_weights[layer]+eps)/(gsum_weights[layer]+eps)) * gw
            layer.w += dx
            xsum_weights[layer] = rho*xsum_weights[layer] + (1-rho)*dx*dx

            gb = der_biases[layer] /batch_len
            gsum_biases[layer]  = rho*gsum_biases[layer]  + (1-rho)*gb*gb
            dx = -np.sqrt((xsum_biases[layer] +eps)/(gsum_biases[layer] +eps)) * gb
            layer.b += dx
            xsum_biases[layer]  = rho*xsum_biases[layer]  + (1-rho)*dx*dx
