import abc

import numpy as np


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        raise AssertionError


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        for _, layer in layers:
            gw = sum_der_w[layer]/batch_len
            layer.w += -(self.lr*gw)

            gb = sum_der_b[layer]/batch_len
            layer.b += -(self.lr*gb)


class Adadelta(Optimizer):
    def __init__(self):
        self.rho = 0.95
        self.eps = 1e-8

    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        gsum_w = {layer: 0 for _, layer in layers}
        xsum_w = {layer: 0 for _, layer in layers}
        gsum_b = {layer: 0 for _, layer in layers}
        xsum_b = {layer: 0 for _, layer in layers}

        for _, layer in layers:
            gw = sum_der_w[layer]/batch_len
            gsum_w[layer] = rho*gsum_w[layer] + (1-rho)*gw*gw
            dx = -np.sqrt((xsum_w[layer]+eps)/(gsum_w[layer]+eps)) * gw
            layer.w += dx
            xsum_w[layer] = rho*xsum_w[layer] + (1-rho)*dx*dx

            gb = sum_der_b[layer] /batch_len
            gsum_b[layer] = rho*gsum_b[layer] + (1-rho)*gb*gb
            dx = -np.sqrt((xsum_b[layer]+eps)/(gsum_b[layer]+eps)) * gb
            layer.b += dx
            xsum_b[layer] = rho*xsum_b[layer] + (1-rho)*dx*dx
