import abc

import numpy as np

import functions as f
import utils as u


class Layer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.depth  = None
        self.height = None
        self.width  = None
        self.n_out  = None
        self.w      = None
        self.b      = None

    @abc.abstractmethod
    def connect_to(self, prev_layer):
        raise AssertionError

    @abc.abstractmethod
    def feedforward(self, prev_layer):
        raise AssertionError

    @abc.abstractmethod
    def backpropagate(self, prev_layer, delta_z):
        raise AssertionError


class InputLayer(Layer):
    def __init__(self, height, width):
        super().__init__()
        self.depth  = 1
        self.height = height
        self.width  = width
        self.n_out  = self.depth*self.height*self.width

    def connect_to(self, prev_layer):
        raise AssertionError

    def feedforward(self, prev_layer):
        raise AssertionError

    def backpropagate(self, prev_layer, delta_z):
        raise AssertionError


class FullyConnectedLayer(Layer):
    def __init__(self, height, init_func, act_func):
        super().__init__()
        self.depth  = 1
        self.height = height
        self.width  = 1
        self.n_out  = self.depth*self.height*self.width
        self.init_func = init_func
        self.act_func = act_func
        self.der_act_func = getattr(f, "der_%s" % act_func.__name__)

    def connect_to(self, prev_layer):
        self.w = self.init_func((self.n_out, prev_layer.n_out),
            prev_layer.n_out, self.n_out)
        self.b = f.zero((self.n_out, 1))

    def feedforward(self, prev_layer):
        """
        Feedforward the observation through the layer

        :param prev_layer: the previous layer of the network
        """
        input_a = prev_layer.a.reshape((prev_layer.a.size, 1))
        self.z = (self.w @ input_a) + self.b

        self.a = self.act_func(self.z)
        assert self.z.shape == self.a.shape

    def backpropagate(self, prev_layer, delta_z):
        """
        Backpropagate the error through the layer

        :param prev_layer: the previous layer of the network
        :param delta_z: the error propagated backward by the next layer of the network
        :returns: the amount of change of input weights of this layer, the amount of change of the biases of this layer
            and the error propagated by this layer
        """
        assert delta_z.shape == self.z.shape == self.a.shape

        input_a = prev_layer.a.reshape((prev_layer.a.size, 1))
        der_w = delta_z @ input_a.T

        der_b = np.copy(delta_z)

        delta_zl = (self.w.T @ delta_z).reshape(prev_layer.z.shape) * self.der_act_func(prev_layer.z)

        return der_w, der_b, delta_zl


class ConvolutionalLayer(Layer):
    def __init__(self, depth, kernel_size, init_func, act_func):
        super().__init__()
        self.depth = depth
        self.kernel_size = kernel_size
        self.init_func = init_func
        self.act_func = act_func
        self.der_act_func = getattr(f, "der_%s" % act_func.__name__)

    def connect_to(self, prev_layer):
        stride_length = 1
        self.height = ((prev_layer.height-self.kernel_size) // stride_length) + 1
        self.width  = ((prev_layer.width -self.kernel_size) // stride_length) + 1
        self.n_out  = self.depth*self.height*self.width

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

        filters_c_out = self.w.shape[0]
        filters_c_in  = self.w.shape[1]
        filters_h     = self.w.shape[2]
        filters_w     = self.w.shape[3]

        image_c = prev_layer.a.shape[0]
        assert image_c == filters_c_in
        image_h = prev_layer.a.shape[1]
        image_w = prev_layer.a.shape[2]

        stride = 1
        new_h = ((image_h-filters_h) // stride) + 1
        new_w = ((image_w-filters_w) // stride) + 1

        self.z = np.zeros((filters_c_out, new_h, new_w))
        for fc in range(filters_c_out):
            src = prev_layer.a
            dst = self.z[fc]
            for ic in range(image_c):
                flt = self.w[fc, ic]
                for i, m in enumerate(range(image_h)):
                    for j, n in enumerate(range(image_w)):
                        src_window = src[ic, m:m+filters_h, n:n+filters_w]
                        if src_window.shape != flt.shape:
                            # out of borders
                            break
                        dst[i, j] += np.convolve(src_window.ravel(), flt.ravel(), mode="valid")

        for r in range(self.depth):
            self.z[r] += self.b[r]

        self.a = np.vectorize(self.act_func)(self.z)
        assert self.a.shape == self.z.shape

    def backpropagate(self, prev_layer, delta_z):
        """
        Backpropagate the error through the layer

        :param prev_layer: the previous layer of the network. The activations of the previous layer are a list of
            feature maps, where each feature map is a 2d matrix
        :param delta_z:
        """
        assert delta_z.shape[0] == self.depth

        der_w = np.empty_like(self.w)
        for t in range(prev_layer.depth):
            for r in range(self.depth):
                src = prev_layer.a[t]
                err =      delta_z[r]
                dst =  der_w[r, t]
                for h in range(self.kernel_size):
                    for v in range(self.kernel_size):
                        src_window = src[v:self.height, h:self.width]
                        err_window = err[v:self.height, h:self.width]
                        dst[h, v] = np.sum(src_window * err_window)

        der_b = np.empty((self.depth, 1))
        for r in range(self.depth):
            der_b[r] = np.sum(delta_z[r])

        delta_zl = np.zeros_like(prev_layer.a)
        for t in range(prev_layer.depth):
            for r in range(self.depth):
                src    =    delta_z[r]
                kernel =     self.w[r, t]
                dst    =   delta_zl[t]
                for i, m in enumerate(range(0, prev_layer.height, self.kernel_size)):
                    for j, n in enumerate(range(0, prev_layer.width, self.kernel_size)):
                        dst_window = dst[m:m+self.kernel_size, n:n+self.kernel_size]
                        rows, cols = dst_window.shape
                        dst_window += kernel[:rows, :cols] * src[i, j]

        return der_w, der_b, delta_zl


class MaxPoolingLayer(Layer):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def connect_to(self, prev_layer):
        assert isinstance(prev_layer, ConvolutionalLayer)
        self.depth  = prev_layer.depth
        stride_length = self.pool_size
        self.height = ((prev_layer.height-self.pool_size) // stride_length) + 1
        self.width  = ((prev_layer.width -self.pool_size) // stride_length) + 1
        self.n_out  = self.depth*self.height*self.width

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

        prev_layer_fmap_size = prev_layer.height
        assert prev_layer_fmap_size % self.pool_size == 0

        self.z = np.zeros((self.depth, self.height, self.width))
        for t, r in zip(range(prev_layer.depth), range(self.depth)):
            src = prev_layer.a[t]
            dst =       self.z[r]
            for i, m in enumerate(range(0, prev_layer.height, self.pool_size)):
                for j, n in enumerate(range(0, prev_layer.width, self.pool_size)):
                    src_window = src[m:m+self.pool_size, n:n+self.pool_size]
                    assert src_window.shape == (self.pool_size, self.pool_size)
                    # downsampling
                    dst[i, j] = np.max(src_window)

        self.a = self.z

    def backpropagate(self, prev_layer, delta_z):
        """
        Backpropagate the error through the layer. Given any pair source(convolutional)/destination(pooling) feature
        maps, each unit of the destination feature map propagates an error to a window (self.pool_size, self.pool_size)
        of the source feature map

        :param prev_layer: the previous layer of the network
        :param delta_z: a tensor of shape (self.depth, self.height, self.width)
        """
        assert self.w.size == 0
        assert isinstance(prev_layer, ConvolutionalLayer)
        assert prev_layer.depth == self.depth
        assert prev_layer.a.ndim == 3
        assert delta_z.shape == (self.depth, self.height, self.width)

        der_w = np.array([])

        der_b = np.array([])

        delta_zl = np.kron(delta_z, np.zeros((self.pool_size, self.pool_size)))
        assert delta_zl.shape == (prev_layer.depth, prev_layer.height, prev_layer.width)
        for t, r in zip(range(prev_layer.depth), range(self.depth)):
            src = prev_layer.a[t]
            err =      delta_z[t]
            dst =     delta_zl[r]
            for i, m in enumerate(range(0, prev_layer.height, self.pool_size)):
                for j, n in enumerate(range(0, prev_layer.width, self.pool_size)):
                    src_window = src[m:m+self.pool_size, n:n+self.pool_size]
                    dst_window = dst[m:m+self.pool_size, n:n+self.pool_size]
                    assert src_window.shape == dst_window.shape == (self.pool_size, self.pool_size)
                    # upsampling: the unit which was the max at the forward propagation
                    # receives all the error at backward propagation
                    assert not np.any(dst_window)
                    dst_window[np.unravel_index(src_window.argmax(), src_window.shape)] = err[i, j]

        return der_w, der_b, delta_zl
