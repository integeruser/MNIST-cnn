import abc

import numpy as np

import functions


class Layer(metaclass=abc.ABCMeta):
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.num_neurons_out = width * height * depth
        self.z = None
        self.a = None

    @abc.abstractmethod
    def feedforward(self):
        raise AssertionError

    @abc.abstractmethod
    def backpropagate(self):
        raise AssertionError


###############################################################################

class InputLayer(Layer):
    def __init__(self, width, height):
        super().__init__(width, height, depth=1)

    def feedforward(self):
        raise AssertionError

    def backpropagate(self):
        raise AssertionError


###############################################################################

class FullyConnectedLayer(Layer):
    def __init__(self, width, height, act_func):
        super().__init__(width, height, depth=1)
        self.act_func = act_func
        self.der_act_func = functions.get_derivative(act_func)

    def feedforward(self, input_a, input_w, input_b):
        """
        Feedforward the observation through the layer

        :param input_a: the activations of the previous layer
        :param input_w: the weights connecting the previous layer with this one
        :param input_b: the biases of this layer
        """
        self.z = input_w @ input_a + input_b
        self.a = self.act_func(self.z)

    def backpropagate(self, input_z, input_a, input_w, delta_zlp):
        """
        Backpropagate the error through the layer

        :param input_z: the zetas of the previous layer
        :param input_a: the activations of the previous layer
        :param input_w: the weights connecting the previous layer with this one
        :param delta_zlp: the error propagated by the following layer
        :returns: the amount of change of input weights of this layer, the amount of change of the biases of this layer
            and the error propagated by this layer
        """
        # compute the derivatives of the weights and biases
        d_der_w = delta_zlp @ input_a.T
        d_der_b = delta_zlp
        # backpropagate the error for the previous layer
        delta_zl = input_w.T @ delta_zlp * self.der_act_func(input_z)
        return d_der_w, delta_zlp, delta_zl
