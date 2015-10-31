import abc

import numpy as np

import functions


class Layer(metaclass=abc.ABCMeta):
    def __init__(self, depth, height, width):
        self.depth = depth
        self.height = height
        self.width = width
        self.num_neurons_out = depth * height * width

    @abc.abstractmethod
    def feedforward(self, prev_layer, input_w, input_b):
        raise AssertionError

    @abc.abstractmethod
    def backpropagate(self, prev_layer, input_w, delta_z):
        raise AssertionError


###############################################################################

class InputLayer(Layer):
    def __init__(self, height, width):
        super().__init__(1, height, width)

    def feedforward(self, prev_layer, input_w, input_b):
        raise AssertionError

    def backpropagate(self, prev_layer, input_w, delta_z):
        raise AssertionError


###############################################################################

class FullyConnectedLayer(Layer):
    def __init__(self, width, height, act_func):
        super().__init__(1, height, width)
        self.act_func = act_func
        self.der_act_func = functions.get_derivative(act_func)

    def feedforward(self, prev_layer, input_w, input_b):
        """
        Feedforward the observation through the layer

        :param prev_layer: the previous layer of the network
        :param input_w: the weights connecting the previous layer with this one
        :param input_b: the biases of this layer
        """
        input_a = prev_layer.a.reshape((prev_layer.a.size, 1))
        self.z = (input_w @ input_a) + input_b
        self.a = self.act_func(self.z)
        assert self.z.shape == self.a.shape

    def backpropagate(self, prev_layer, input_w, delta_z):
        """
        Backpropagate the error through the layer

        :param prev_layer: the previous layer of the network
        :param input_w: the weights connecting the previous layer with this one
        :param delta_z: the error propagated backward by the next layer of the network
        :returns: the amount of change of input weights of this layer, the amount of change of the biases of this layer
            and the error propagated by this layer
        """
        input_a = prev_layer.a.reshape((prev_layer.a.size, 1))
        # compute the derivatives of the weights and biases
        der_input_w = delta_z @ input_a.T
        der_input_b = delta_z
        # backpropagate the error for the previous layer
        assert prev_layer.z.shape == prev_layer.a.shape
        delta_zl = (input_w.T @ delta_z).reshape(prev_layer.z.shape) * self.der_act_func(prev_layer.z)
        return der_input_w, der_input_b, delta_zl

