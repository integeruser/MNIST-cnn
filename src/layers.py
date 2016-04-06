import abc

import numpy as np

import functions as f
import utils as u


class Layer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.depth = None
        self.height = None
        self.width = None
        self.n_out = None
        self.w = None
        self.b = None

    @abc.abstractmethod
    def connect_to(self, prev_layer):
        raise AssertionError

    @abc.abstractmethod
    def feedforward(self, prev_layer):
        raise AssertionError

    @abc.abstractmethod
    def backpropagate(self, prev_layer, delta):
        raise AssertionError


class InputLayer(Layer):
    def __init__(self, height, width):
        super().__init__()
        self.depth = 1
        self.height = height
        self.width = width
        self.n_out = self.depth * self.height * self.width
        self.der_act_func = lambda x: x

    def connect_to(self, prev_layer):
        raise AssertionError

    def feedforward(self, prev_layer):
        raise AssertionError

    def backpropagate(self, prev_layer, delta):
        raise AssertionError


class FullyConnectedLayer(Layer):
    def __init__(self, height, init_func, act_func):
        super().__init__()
        self.depth = 1
        self.height = height
        self.width = 1
        self.n_out = self.depth * self.height * self.width
        self.init_func = init_func
        self.act_func = act_func
        self.der_act_func = getattr(f, "der_%s" % act_func.__name__)

    def connect_to(self, prev_layer):
        self.w = self.init_func((self.n_out, prev_layer.n_out), prev_layer.n_out, self.n_out)
        self.b = f.zero((self.n_out, 1))

    def feedforward(self, prev_layer):
        """
        Feedforward the observation through the layer

        :param prev_layer: the previous layer of the network
        """
        prev_a = prev_layer.a.reshape((prev_layer.a.size, 1))

        self.z = (self.w @ prev_a) + self.b

        self.a = self.act_func(self.z)
        assert self.z.shape == self.a.shape

    def backpropagate(self, prev_layer, delta):
        """
        Backpropagate the error through the layer

        :param prev_layer: the previous layer of the network
        :param delta: the error propagated backward by the next layer of the network
        :returns: the amount of change of input weights of this layer, the amount of change of the biases of this layer
            and the error propagated by this layer
        """
        assert delta.shape == self.z.shape == self.a.shape

        prev_a = prev_layer.a.reshape((prev_layer.a.size, 1))

        der_w = delta @ prev_a.T

        der_b = np.copy(delta)

        prev_delta = (self.w.T @ delta).reshape(prev_layer.z.shape) * prev_layer.der_act_func(prev_layer.z)

        return der_w, der_b, prev_delta


class ConvolutionalLayer(Layer):
    def __init__(self, depth, kernel_size, init_func, act_func):
        super().__init__()
        self.depth = depth
        self.kernel_size = kernel_size
        self.init_func = init_func
        self.act_func = act_func
        self.der_act_func = getattr(f, "der_%s" % act_func.__name__)

    def connect_to(self, prev_layer):
        self.stride_length = 1
        self.height = ((prev_layer.height - self.kernel_size) // self.stride_length) + 1
        self.width  = ((prev_layer.width  - self.kernel_size) // self.stride_length) + 1
        self.n_out = self.depth * self.height * self.width

        self.w = self.init_func((self.depth, prev_layer.depth, self.kernel_size, self.kernel_size),
            prev_layer.n_out, self.n_out)
        self.b = f.zero((self.depth, 1))

    def feedforward(self, prev_layer):
        """
        Feedforward the observation through the layer

        :param prev_layer: the previous layer of the network. The activations of the previous layer must be a list of
            feature maps, where each feature map is a 2d matrix
        """
        assert self.w.shape == (self.depth, prev_layer.depth, self.kernel_size, self.kernel_size)
        assert self.b.shape == (self.depth, 1)
        assert prev_layer.a.ndim == 3

        prev_a = prev_layer.a

        filters_c_out = self.w.shape[0]
        filters_c_in = self.w.shape[1]
        filters_h = self.w.shape[2]
        filters_w = self.w.shape[3]

        image_c = prev_a.shape[0]
        assert image_c == filters_c_in
        image_h = prev_a.shape[1]
        image_w = prev_a.shape[2]

        stride = 1
        new_h = ((image_h - filters_h) // stride) + 1
        new_w = ((image_w - filters_w) // stride) + 1

        self.z = np.zeros((filters_c_out, new_h, new_w))
        for r in range(filters_c_out):
            for t in range(image_c):
                filter = self.w[r, t]
                for i, m in enumerate(range(0, image_h - filters_h + 1, self.stride_length)):
                    for j, n in enumerate(range(0, image_w - filters_w + 1, self.stride_length)):
                        prev_a_window = prev_a[t, m:m+filters_h, n:n+filters_w]
                        self.z[r, i, j] += np.correlate(prev_a_window.ravel(), filter.ravel(), mode="valid")

        for r in range(self.depth):
            self.z[r] += self.b[r]

        self.a = np.vectorize(self.act_func)(self.z)
        assert self.a.shape == self.z.shape

    def backpropagate(self, prev_layer, delta):
        """
        Backpropagate the error through the layer

        :param prev_layer: the previous layer of the network. The activations of the previous layer are a list of
            feature maps, where each feature map is a 2d matrix
        :param delta:
        """
        assert delta.shape[0] == self.depth

        prev_a = prev_layer.a

        der_w = np.empty_like(self.w)
        for r in range(self.depth):
            for t in range(prev_layer.depth):
                for h in range(self.kernel_size):
                    for v in range(self.kernel_size):
                        prev_a_window = prev_a[t, v:v+self.height-self.kernel_size+1:self.stride_length,
                                                  h:h+self.width -self.kernel_size+1:self.stride_length]
                        delta_window  =  delta[r, v:v+self.height-self.kernel_size+1:self.stride_length,
                                                  h:h+self.width -self.kernel_size+1:self.stride_length]
                        assert prev_a_window.shape == delta_window.shape
                        der_w[r, t, h, v] = np.sum(prev_a_window * delta_window)

        der_b = np.empty((self.depth, 1))
        for r in range(self.depth):
            der_b[r] = np.sum(delta[r])

        prev_delta = np.zeros_like(prev_a)
        for r in range(self.depth):
            for t in range(prev_layer.depth):
                kernel = self.w[r, t]
                for i, m in enumerate(range(0, prev_layer.height - self.kernel_size + 1, self.stride_length)):
                    for j, n in enumerate(range(0, prev_layer.width - self.kernel_size + 1, self.stride_length)):
                        prev_delta[t, m:m+self.kernel_size, n:n+self.kernel_size] += kernel * delta[r, i, j]
        prev_delta *= prev_layer.der_act_func(prev_layer.z)

        return der_w, der_b, prev_delta


class MaxPoolingLayer(Layer):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size
        self.der_act_func = lambda x: x

    def connect_to(self, prev_layer):
        assert isinstance(prev_layer, ConvolutionalLayer)
        self.depth = prev_layer.depth
        self.height = ((prev_layer.height - self.pool_size) // self.pool_size) + 1
        self.width  = ((prev_layer.width  - self.pool_size) // self.pool_size) + 1
        self.n_out = self.depth * self.height * self.width

        self.w = np.empty((0))
        self.b = np.empty((0))

    def feedforward(self, prev_layer):
        """
        Feedforward the observation through the layer

        :param prev_layer: the previous layer of the network. The activations of the previous layer must be a list of
            feature maps, where each feature map is a 3d matrix
        """
        assert self.w.size == 0
        assert self.b.size == 0
        assert isinstance(prev_layer, ConvolutionalLayer)
        assert prev_layer.depth == self.depth
        assert prev_layer.a.ndim == 3

        prev_a = prev_layer.a

        prev_layer_fmap_size = prev_layer.height
        assert prev_layer_fmap_size % self.pool_size == 0

        self.z = np.zeros((self.depth, self.height, self.width))
        for r, t in zip(range(self.depth), range(prev_layer.depth)):
            assert r == t
            for i, m in enumerate(range(0, prev_layer.height, self.pool_size)):
                for j, n in enumerate(range(0, prev_layer.width, self.pool_size)):
                    prev_a_window = prev_a[t, m:m+self.pool_size, n:n+self.pool_size]
                    assert prev_a_window.shape == (self.pool_size, self.pool_size)
                    # downsampling
                    self.z[r, i, j] = np.max(prev_a_window)

        self.a = self.z

    def backpropagate(self, prev_layer, delta):
        """
        Backpropagate the error through the layer. Given any pair source(convolutional)/destination(pooling) feature
        maps, each unit of the destination feature map propagates an error to a window (self.pool_size, self.pool_size)
        of the source feature map

        :param prev_layer: the previous layer of the network
        :param delta: a tensor of shape (self.depth, self.height, self.width)
        """
        assert self.w.size == 0
        assert self.b.size == 0
        assert isinstance(prev_layer, ConvolutionalLayer)
        assert prev_layer.depth == self.depth
        assert prev_layer.a.ndim == 3
        assert delta.shape == (self.depth, self.height, self.width)

        prev_a = prev_layer.a

        der_w = np.array([])

        der_b = np.array([])

        prev_delta = np.empty_like(prev_a)
        for r, t in zip(range(self.depth), range(prev_layer.depth)):
            assert r == t
            for i, m in enumerate(range(0, prev_layer.height, self.pool_size)):
                for j, n in enumerate(range(0, prev_layer.width, self.pool_size)):
                    prev_a_window = prev_a[t, m:m+self.pool_size, n:n+self.pool_size]
                    assert prev_a_window.shape == (self.pool_size, self.pool_size)
                    # upsampling: the unit which was the max at the forward propagation
                    # receives all the error at backward propagation (the other units receive zero)
                    max_unit_index = np.unravel_index(prev_a_window.argmax(), prev_a_window.shape)
                    prev_delta_window = np.zeros_like(prev_a_window)
                    prev_delta_window[max_unit_index] = delta[t, i, j]
                    prev_delta[r, m:m+self.pool_size, n:n+self.pool_size] = prev_delta_window

        return der_w, der_b, prev_delta
