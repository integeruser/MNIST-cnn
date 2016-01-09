import abc

import scipy.signal
import numpy as np

import functions as f


class Layer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.depth  = None
        self.height = None
        self.width  = None

    @abc.abstractmethod
    def connect_to(self, prev_layer):
        raise AssertionError

    @abc.abstractmethod
    def feedforward(self, prev_layer, w, b):
        raise AssertionError

    @abc.abstractmethod
    def backpropagate(self, prev_layer, w, delta_z):
        raise AssertionError


class InputLayer(Layer):
    def __init__(self, height, width):
        super().__init__()
        self.depth  = 1
        self.height = height
        self.width  = width

    def connect_to(self, prev_layer):
        raise AssertionError

    def feedforward(self, prev_layer, w, b):
        raise AssertionError

    def backpropagate(self, prev_layer, w, delta_z):
        raise AssertionError


class FullyConnectedLayer(Layer):
    def __init__(self, height, act_func):
        super().__init__()
        self.depth  = 1
        self.height = height
        self.width  = 1
        self.act_func = act_func
        self.der_act_func = getattr(f, "der_%s" % act_func.__name__)

    def connect_to(self, prev_layer):
        pass

    def feedforward(self, prev_layer, w, b):
        """
        Feedforward the observation through the layer

        :param prev_layer: the previous layer of the network
        :param w: the weights connecting the previous layer with this one
        :param b: the biases of this layer
        """
        input_a = prev_layer.a.reshape((prev_layer.a.size, 1))
        self.z = (w @ input_a) + b

        self.a = self.act_func(self.z)
        assert self.z.shape == self.a.shape

    def backpropagate(self, prev_layer, w, delta_z):
        """
        Backpropagate the error through the layer

        :param prev_layer: the previous layer of the network
        :param w: the weights connecting the previous layer with this one
        :param delta_z: the error propagated backward by the next layer of the network
        :returns: the amount of change of input weights of this layer, the amount of change of the biases of this layer
            and the error propagated by this layer
        """
        assert delta_z.shape == self.z.shape == self.a.shape

        input_a = prev_layer.a.reshape((prev_layer.a.size, 1))
        der_w = delta_z @ input_a.T

        der_b = np.copy(delta_z)

        delta_zl = (w.T @ delta_z).reshape(prev_layer.z.shape) * self.der_act_func(prev_layer.z)

        return der_w, der_b, delta_zl


class ConvolutionalLayer(Layer):
    def __init__(self, depth, kernel_size, act_func):
        super().__init__()
        self.depth = depth

        self.kernel_size = kernel_size
        self.stride_length = 1
        self.act_func = act_func
        self.der_act_func = getattr(f, "der_%s" % act_func.__name__)

    def connect_to(self, prev_layer):
        self.height = (prev_layer.height-self.kernel_size) // self.stride_length + 1
        self.width  = (prev_layer.width -self.kernel_size) // self.stride_length + 1

    def feedforward(self, prev_layer, w, b):
        """
        Feedforward the observation through the layer

        :param prev_layer: the previous layer of the network. The activations of the previous layer must be a list of
            feature maps, where each feature map is a 2d matrix
        :param w: the weights connecting the previous layer with this one (must be a list of 3d tensors with shape
            (depth, prev_layer.depth, kernel_size, kernel_size))
        :param b: the biases of this layer (must be a list of scalars with shape (depth, 1))
        """
        assert prev_layer.a.ndim == 3
        assert w.shape == (self.depth, prev_layer.depth, self.kernel_size, self.kernel_size)
        assert b.shape == (self.depth, 1)
        self.z = np.array([scipy.signal.convolve(prev_layer.a, fmap, mode="valid") for fmap in w])
        for r in range(self.depth):
            self.z[r] += b[r]

        self.a = np.vectorize(self.act_func)(self.z)
        assert self.z.shape == self.a.shape

    def backpropagate(self, prev_layer, w, delta_z):
        """
        Backpropagate the error through the layer

        :param prev_layer: the previous layer of the network. The activations of the previous layer are a list of
            feature maps, where each feature map is a 2d matrix
        :param w: the weights connecting the previous layer with this one (a list of tensors with shape
            (depth, prev_layer.depth, kernel_size, kernel_size))
        :param delta_z:
        """
        assert delta_z.shape[0] == self.depth

        der_w = np.empty_like(w)
        for t in range(prev_layer.depth):
            for r in range(self.depth):
                src = prev_layer.a[t]
                err =      delta_z[r, t]
                dst =  der_w[r, t]
                for h in range(self.kernel_size):
                    for v in range(self.kernel_size):
                        src_window = src[v:self.height:self.stride_length, h:self.width:self.stride_length]
                        err_window = err[v:self.height:self.stride_length, h:self.width:self.stride_length]
                        dst[h, v] = np.sum(src_window * err_window)

        der_b = np.empty((self.depth, 1))
        for r in range(self.depth):
            der_b[r] = np.sum(delta_z[r])

        delta_zl = np.zeros_like(prev_layer.a)
        for t in range(prev_layer.depth):
            for r in range(self.depth):
                src =    delta_z[r, t]
                kernel = w[r, t]
                dst =    delta_zl[t]
                for m in range(0, prev_layer.height, self.kernel_size):
                    for n in range(0, prev_layer.width, self.kernel_size):
                        dst_window = dst[m:m+self.kernel_size, n:n+self.kernel_size]
                        i = m // self.kernel_size
                        j = n // self.kernel_size
                        rows, cols = dst_window.shape
                        dst_window += kernel[:rows, :cols] * src[i, j]

        return der_w, der_b, delta_zl


class MaxPoolingLayer(Layer):
    def __init__(self, window_size):
        super().__init__()

        self.window_size = window_size
        self.stride_length = window_size

    def connect_to(self, prev_layer):
        assert isinstance(prev_layer, ConvolutionalLayer)
        self.depth  = prev_layer.depth
        self.height = (prev_layer.height-self.window_size) // self.stride_length + 1
        self.width  = (prev_layer.width -self.window_size) // self.stride_length + 1


    def feedforward(self, prev_layer, w, b):
        """
        Feedforward the observation through the layer

        :param prev_layer: the previous layer of the network. The activations of the previous layer must be a list of
            feature maps, where each feature map is a 3d matrix
        :param w: should be an empty array (no weights between a convolutional layer and a pooling layer)
        :param b: should be an empty array (no biases in a pooling layer)
        """
        assert isinstance(prev_layer, ConvolutionalLayer)
        assert prev_layer.depth == self.depth
        assert prev_layer.a.ndim == 4
        assert w.size == 0
        assert b.size == 0

        prev_layer_fmap_size = prev_layer.height
        assert prev_layer_fmap_size % self.window_size == 0

        self.z = np.zeros((self.depth, self.height, self.width))
        for t, r in zip(range(prev_layer.depth), range(self.depth)):
            src = prev_layer.a[t, 0]
            dst =       self.z[r]
            for m in range(0, prev_layer.height, self.window_size):
                for n in range(0, prev_layer.width, self.window_size):
                    src_window = src[m:m+self.window_size, n:n+self.window_size]
                    assert src_window.shape == (self.window_size, self.window_size)
                    i = m // self.window_size
                    j = n // self.window_size
                    # max() pooling
                    dst[i, j] = np.max(src_window)

        self.a = self.z

    def backpropagate(self, prev_layer, w, delta_z):
        """
        Backpropagate the error through the layer. Given any pair source(convolutional)/destination(pooling) feature
        maps, each unit of the destination feature map propagates an error to a window (self.window_size, self.window_size)
        of the source feature map

        :param prev_layer: the previous layer of the network
        :param w: the weights connecting the previous layer with this one. Since the previous layer is for sure
            a convolutional layer, this tensor should be empty
        :param delta_z: a tensor of shape (self.depth, self.height, self.width)
        """
        assert isinstance(prev_layer, ConvolutionalLayer)
        assert prev_layer.depth == self.depth
        assert prev_layer.a.ndim == 4
        assert w.size == 0
        assert delta_z.shape == (self.depth, self.height, self.width)

        der_w = np.array([])

        der_b = np.array([])

        delta_zl = np.kron(delta_z, np.ones((self.window_size, self.window_size)))
        delta_zl = np.expand_dims(delta_zl, axis=1)
        assert delta_zl.shape == (prev_layer.depth, 1, prev_layer.height, prev_layer.width)
        for t, r in zip(range(prev_layer.depth), range(self.depth)):
            src = prev_layer.a[t, 0]
            dst =     delta_zl[r, 0]
            for m in range(0, prev_layer.height, self.window_size):
                for n in range(0, prev_layer.width, self.window_size):
                    src_window = src[m:m+self.window_size, n:n+self.window_size]
                    dst_window = dst[m:m+self.window_size, n:n+self.window_size]
                    assert src_window.shape == dst_window.shape == (self.window_size, self.window_size)
                    # derivative of max()
                    der = np.zeros_like(src_window, dtype=np.uint8)
                    der[np.unravel_index(src_window.argmax(), src_window.shape)] = 1
                    dst_window[:] *= der

        return der_w, der_b, delta_zl
