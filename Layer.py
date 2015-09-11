__author__ = 'F. Cagnin and A. Torcinovich'

import abc

import numpy as np

import polling_functions
import act_functions


class Layer(object):
    """
    NAME
        Layer

    DESCRIPTION
        This class represents a general network layer, which must specify its size, and must implement a feedforward and
        backpropagation method

    ATTRIBUTES
    size (abstract property):   the size of the layer. It can have different meaning, depending on the nature of the
                                layer
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def size(self):
        raise NotImplementedError

    @size.getter
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    @abc.abstractmethod
    def feedforward(self, a, w, b):
        """feed forward the observation in the current layer"""
        raise NotImplementedError

    @abc.abstractmethod
    def backpropagate(self, z, a, w, delta_zlp):
        """backpropagate the error in the current layer"""
        raise NotImplementedError


class InputLayer(Layer):
    def __init__(self, size):
        self.size = size
        self.num_neurons = size ** 2

    def feedforward(self, a, w, b):
        raise NotImplementedError

    def backpropagate(self, z, a, w, delta_zlp):
        raise NotImplementedError


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

    def __init__(self, size, act_func_str):
        """Initialize the layer.
        :param size: the number of neurons of the layer
        :param act_func_str: string representing te activation function (see NeuralNetwork.__init__), used to retrieve
                             the derivative too
        """
        self.size = size
        self.num_neurons = size
        self.act_func = getattr(act_functions, act_func_str)
        self.der_act_func = getattr(act_functions, 'der_' + act_func_str)

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


class ConvolutionalLayer(Layer):
    """
    NAME
        ConvolutionalLayer

    DESCRIPTION
        This class implements a convolutional layer

    ATTRIBUTES
        self.size:          the side size of the layer (the square root of the number of neurons)
        self.kernel_size:   the side size of the kernel
        self.stride_length: the shifting amount of the kernel
        self.poll_func:      the activation function
        self.der_poll_func:  the derivative of the activation function
        self.prev_layer:    a reference to the previous layer in a network needed for certain polling functions
    """

    def __init__(self, size, kernel_size, stride_length, act_func_str):
        """
        :param size:  the side size of the layer (the square root of the number of neurons)
        :param kernel_size: the side size of the kernel
        :param stride_length: the shifting amount of the kernel
        :param act_func_str: string representing the activation function (see NeuralNetwork.__init__), used to retrieve
                             the derivative too
        """
        self.size = size
        self.num_neurons = size ** 2
        self.kernel_size = kernel_size
        self.stride_length = stride_length
        self.act_func = getattr(act_functions, act_func_str)
        self.der_act_func = getattr(act_functions, 'der_' + act_func_str)

    def feedforward(self, a, w, b):
        """ feedforward the observation through the layer
        :param a: the activations of the previous layer
        :param w: the weights connecting this layer with the previous one
        :param b: the bias associated to this layer
        :return new_z: the zetas of this layer
                new_a: the activations of this layer
        """
        new_z = np.zeros((self.size, self.size))
        int_i = int_j = np.array(range(0, self.size))
        for i1, i2 in zip(int_i * self.stride_length, int_i):
            for j1, j2 in zip(int_j * self.stride_length, int_j):
                new_z[i2, j2] = sum(sum(w * a[i1:(i1 + self.kernel_size), j1:(j1 + self.kernel_size)])) + b
        new_a = self.act_func(new_z)
        return new_z, new_a

    def backpropagate(self, z, a, w, delta_zlp):
        """backpropagate the error through the layer
        :param z: the zetas of this layer
        :param a: the activations of this layer
        :param w: the weights associated to this layer and the previous one
        :param delta_zlp: the error propagated by the previous layer
        :return d_der_w: the amount of change of the weights under consideration
                d_der_b: the amount of change of the biases under consideration
                delta_zl:  the error propagated by this layer
        """
        delta_zlp = np.reshape(delta_zlp, (self.size, self.size))
        d_der_w = np.zeros((self.kernel_size, self.kernel_size))
        interval = np.array(range(0, self.size)) * self.stride_length
        int_i1, int_j1 = map(list, zip(*[(p1, p2) for p1 in interval for p2 in interval]))
        # int_i1 = [p[0] for p in pairs]
        # int_j1 = [p[1] for p in pairs]
        for iw in range(0, self.kernel_size):
            for jw in range(0, self.kernel_size):
                d_der_w[iw, jw] = sum(sum(delta_zlp * np.reshape(a[[i + iw for i in int_i1], [j + jw for j in int_j1]],
                                                                 (self.size, self.size))))
        # TODO is it a correct expression for d_der_b?
        d_der_b = sum(sum(delta_zlp))

        size_prev = (self.size - 1) * self.stride_length + self.kernel_size
        delta_zl = np.zeros((size_prev, size_prev))
        for e in range(0, size_prev):
            for f in range(0, size_prev):
                for c in range(0, (size_prev - self.kernel_size) // self.stride_length + 1):
                    for d in range(0, (size_prev - self.kernel_size) // self.stride_length + 1):
                        iw = e - c * self.stride_length
                        jw = f - d * self.stride_length
                        if iw >= 0 and iw < self.kernel_size and jw >= 0 and jw < self.kernel_size:
                            delta_zl[e, f] = delta_zl[e, f] + delta_zlp[c, d] * w[iw, jw]

        return d_der_w, d_der_b, delta_zl


class PollingLayer(Layer):
    """This class represents a polling layer"""

    # N.B. in polling layers stride_length = kernel_size
    def __init__(self, size, kernel_size, poll_func_str, add_params):
        self.size = size
        self.num_neurons = size ** 2
        self.kernel_size = kernel_size
        self.poll_func = getattr(polling_functions, poll_func_str)
        self.der_poll_func = getattr(polling_functions, 'der_' + poll_func_str)
        self.add_params = add_params

    def feedforward(self, a, w, b):
        """ feedforward the observation through the layer
        :param a: the activations of the previous layer
        :param w: the weights connecting this layer with the previous one
        :param b: the bias associated to this layer
        :return: new_z: the zetas of this layer
                 new_a: the activations of this layer

        Be aware that in polling layer no activations function is applied, so the activations new_a are equal to new_z.
        """

        new_z = np.zeros((self.size, self.size))
        int_i = int_j = np.array(range(0, self.size))
        for i1, i2 in zip(int_i * self.kernel_size, int_i):
            for j1, j2 in zip(int_j * self.kernel_size, int_j):
                new_z[i2, j2] = self.poll_func(a[i1:(i1 + self.kernel_size), j1:(j1 + self.kernel_size)],
                                               self.add_params)
        return new_z, new_z

    def backpropagate(self, z, a, w, delta_zlp):
        """backpropagate the error through the layer
        :param z: the zetas of this layer
        :param a: the activations of this layer
        :param w: the weights associated to this layer and the previous one
        :param delta_zlp: the error propagated by the previous layer
        :return d_der_w: the amount of change of the weights under consideration
                delta_zlp: the amount of change of the biases under consideration
                delta_zl:  the error propagated by this layer

        Be aware that no weights and biases are related to polling layer, so NaN is returned in place of their amounts
        of change.
        """

        delta_zl = np.kron(np.reshape(delta_zlp, (self.size, self.size)), np.ones((self.kernel_size, self.kernel_size)))
        for c in range(0, self.size):
            for d in range(0, self.size):
                # TODO check this command when you have written the polling functions
                delta_zl[c: (c + self.kernel_size), d: (d + self.kernel_size)] *= \
                    self.der_poll_func(delta_zl[c: (c + self.kernel_size), d: (d + self.kernel_size)], self.add_params)

        # interval = [i * self.kernel_size for i in range(0, self.size)]
        # int_i1,int_j1 = map(list, zip(*[(p1, p2) for p1 in interval for p2 in interval]))
        # # int_i1 = [p[0] for p in pairs]
        # # int_j1 = [p[1] for p in pairs]
        # for iw in range(0, self.kernel_size):
        #     for jw in range(0, self.kernel_size):
        #         delta_zl[iw, jw] = delta_zlp * self.der_poll_func(a[int_i1 + iw, int_j1 + jw])
        return np.NaN, np.NaN, delta_zl
