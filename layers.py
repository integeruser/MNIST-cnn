import abc

import numpy as np

import functions


class Layer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def feedforward(self):
        raise NotImplementedError

    @abc.abstractmethod
    def backpropagate(self):
        raise NotImplementedError


###############################################################################

class InputLayer(Layer):
    pass


class VerticalInputLayer(InputLayer):
    def __init__(self, size):
        self.size = size
        self.num_neurons = size

    def feedforward(self):
        raise NotImplementedError

    def backpropagate(self):
        raise NotImplementedError


class SquaredInputLayer(InputLayer):
    def __init__(self, size):
        self.size = size
        self.num_neurons = size ** 2

    def feedforward(self):
        raise NotImplementedError

    def backpropagate(self):
        raise NotImplementedError


###############################################################################

class FullyConnectedLayer(Layer):
    """
    NAME
        FullyConnectedLayer

    DESCRIPTION
        This class implements a fully connected layer

    ATTRIBUTES
        self.size:          the number of neurons of the layer
        self.act_func:      the activation function
        self.der_act_func:  the derivative of the activation function
        """

    def __init__(self, size, act_func):
        """Initialize the layer.
        :param size: the number of neurons of the layer
        :param act_func: string representing te activation function (see NeuralNetwork.__init__), used to retrieve
                             the derivative too
        """
        self.size = size
        self.num_neurons = size
        self.act_func = act_func
        self.der_act_func = functions.get_derivative(act_func)

    def feedforward(self, a, w, b):
        """ feedforward the observation through the layer
        :param a: the activations of the previous layer
        :param w: the weights connecting this layer with the previous one
        :param b: the bias associated to this layer
        :return new_z: the zetas of this layer
                new_a: the activations of this layer
        """
        a = np.reshape(a, (a.size, 1))
        new_z = np.dot(w, a) + b
        new_a = self.act_func(new_z)
        return new_z, new_a

    def backpropagate(self, z, a, w, delta_zlp):
        """backpropagate the error through the layer
        :param z: the zetas of this layer
        :param a: the activations of this layer
        :param w: the weights associated to this layer and the previous one
        :param delta_zlp: the error propagated by the previous layer
        :return d_der_w: the amount of change of the weights under consideration
                delta_zlp: the amount of change of the biases under consideration
                delta_zl:  the error propagated by this layer
        """
        # compute the derivatives of the weights and biases. N.B. reshape also takes care to transpose a
        d_der_w = np.dot(delta_zlp, np.reshape(a, (1, a.size)))
        # d_der_b = delta_zlp
        # propagate the error for the next layer only if the next layer is not the input one
        delta_zl = np.dot(w.transpose(), delta_zlp) * self.der_act_func(
            np.reshape(z, (z.shape[0] * z.shape[1], 1))) if not isinstance(z, float) else np.NaN
        return d_der_w, delta_zlp, delta_zl
