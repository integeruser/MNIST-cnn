import numpy as np

from layers import InputLayer, FullyConnectedLayer
import functions


class NeuralNetwork():
    """
    NAME
        NeuralNetwork

    DESCRIPTION
        this class represents a general neural network. More specifically shallow, deep, and convolutional neural networks
        can be created using this class.

    ATTRIBUTES
        self.layers:        list containing the layers forming the NN. Each element is a Layer descendants instance (see
                            layers.py).
        self.input_weights:       list containing the input_weights of each layer, according to their nature (for polling layers NaN
                            is stored instead).
        self.input_biases:        list containing the input_biases of each layer, according to their nature (for polling layers NaN
                            is stored instead).
        self.der_cost_func: the derivative of the cost function used for the backpropagation.
        self.zs:            the zetas computed for each layer.
        self.acts:          the activations computed for each layer.
    """

    def __init__(self, layers, cost_func):
        """
        Initialize the data. Initial input_weights and input_biases are chosen randomly, according to a gaussian distribution.
        :param layers_info:    it's the vector which stores the info of each layer of the NN. In order to create a
                    correct NN it must be initialized in this way:
                    1. The first element is the size of the input layer. In case of a CNN the image must be n x n.
                    2. The successive elements must contain one of these three layers:
                        - A Fully Connected Layer: in this case the element is a 3-tuple containing a string with value
                            fcl', the number of elements in the layer (referred as layer size) and a string with the
                            activation function name (look below for the accepted values). Ex. ('fcl',23,'sigmoid')
                        - A Convolutional Layer: in this case the layer will be defined starting from its kernel and the
                            side of the resulting layer (referred as layer size) will be computed accordingly. The
                            element is a 4-tuple containing a string with value 'cl', the kernel side size (referred as
                            kernel size) the shifting of the window (referred as stride length and equal in  both the
                            horizontal and vertical direction) and a string with the activation function name (look
                            below for the accepted values). Ex. ('cl',5,2,'sigmoid'). Be careful to choose the kernel
                            size and stride length such that the kernel correctly fits the layer which is applied to.
                        - A Polling Layer: in this case the layer will be defined starting from its kernel and the side
                            of the layer (referred as layer size) will be computed accordingly. The element is a 3-tuple
                            containing a string with value 'pl', the kernel side size (referred as kernel size), a
                            string with the polling function name (look below for the accepted values) and a list
                            containing additional parameters for the polling functions. Ex. ('pl',2,'mean',[3]). The
                            'stride length' of a polling layer is equal to its kernel size. Be careful to choose the
                            kernel size such that the kernel correctly fits the layer which is applied to.
                    3. The last element must be chosen accordingly. Usually it's a FCL.
        :param cost_func:  the cost functions applied to the net

        - Usually polling layers are chained to convolutional layers, but in our implementation each layer can be
          chained to any layers, without restriction (though the correctness of the results can't be assured).

        - Possible activation function values: 'sigmoid', 'softmax', 'rect_lin'
        - Possible polling function values: 'mean', 'max', 'lp_norm'
        - Possible cost function values: 'quadratic', 'cross_entropy', 'log_likelihood'
        """

        assert len(layers) >= 2
        assert isinstance(layers[0], InputLayer)
        self.input_layer, *self.layers = layers

        self.input_weights = dict()
        self.input_biases = dict()
        prev_layer = self.input_layer
        for layer in self.layers:
            if type(layer) is FullyConnectedLayer:
                self.input_weights[layer] = np.random.normal(0, 1 / np.sqrt(prev_layer.n_out),
                                                             size=(layer.n_out, prev_layer.n_out))
                self.input_biases[layer] = np.random.normal(0, 1 / np.sqrt(prev_layer.n_out),
                                                            size=(layer.n_out, 1))
            else:
                raise NotImplementedError
            prev_layer = layer

        self.der_cost_func = functions.get_derivative(cost_func)

    def feedforward(self, x):
        """
        Perform the feedforward of one observation and return the list of z and activations of each layer
        :param x:           the observation taken as input layer
        :return self.zs:    the list of zetas of each layer
                self.acts:  the list of activations each layer
        """

        # the input layer hasn't zetas and its initial values are considered directly as activations
        zs = {self.input_layer: np.empty(shape=(1, 1))}
        acts = {self.input_layer: x}

        # feedforward the input for each layer
        prev_layer = self.input_layer
        for layer in self.layers:
            w = self.input_weights[layer]
            b = self.input_biases[layer]
            z, a = layer.feedforward(acts[prev_layer], w, b)
            zs[layer] = z
            acts[layer] = a
            prev_layer = layer
        return zs, acts

    def backpropagate(self, batch, eta):
        """
        Perform the backpropagation for one observation and return the derivative of the input_weights relative to each
        layer
        :param y:           the class of the current observation represented in 1-of-k coding
        :return d_der_ws:   list containing the amount of change in the input_weights, due to the current observation
                d_der_bs:   list containing the amount of change in the input_biases, due to the current observation
        """

        der_weights = {layer: np.zeros(self.input_weights[layer].shape) for layer in self.layers}
        der_biases = {layer: np.zeros(self.input_biases[layer].shape) for layer in self.layers}

        # for each observation in the current batch
        for x, y in batch:
            x = np.reshape(x, (x.size, 1))
            y = np.reshape(y, (y.size, 1))

            # feedforward the observation
            zs, acts = self.feedforward(x)

            # backpropagate the error
            output_layer = self.layers[-1]
            delta_zlp = self.der_cost_func(acts[output_layer], y) * output_layer.der_act_func(zs[output_layer])

            d_der_ws = {}
            d_der_bs = {}
            for i, layer in enumerate(reversed(self.layers)):
                j = len(self.layers) - 1 - i - 1
                prev_layer = self.layers[j] if j >= 0 else self.input_layer
                z = zs[prev_layer]
                a = acts[prev_layer]
                w = self.input_weights[layer]
                d_der_w, d_der_b, delta_zlp = layer.backpropagate(z, a, w, delta_zlp)
                d_der_ws[layer] = d_der_w
                d_der_bs[layer] = d_der_b

            # sum the derivatives of the weights and biases of the current observation to the previous ones
            for layer in self.layers:
                der_weights[layer] += d_der_ws[layer]
                der_biases[layer] += d_der_bs[layer]

        # update weights and biases with the results of the current batch
        for layer in self.layers:
            self.input_weights[layer] -= eta / len(batch) * der_weights[layer]
            self.input_biases[layer] -= eta / len(batch) * der_biases[layer]


def train(net, inputs, num_epochs, batch_size, eta):
    """
    Train the network according to the Stochastic Gradient Descent (SGD) algorithm
    :param net:         the network to train
    :param inputs:      the observations that are going to be used to train the network
    :param num_epochs:  the number of epochs
    :param batch_size:  the size of the batch used in single cycle of the SGD
    :param eta:         the learning rate
    """

    assert eta > 0
    assert len(inputs) % batch_size == 0

    for i in range(num_epochs):
        print("Epoch {}".format(i + 1))

        np.random.shuffle(inputs)

        # divide input observations into batches of size batch_size
        batches = [inputs[j:j + batch_size] for j in range(0, len(inputs), batch_size)]
        for batch in batches:
            net.backpropagate(batch, eta)


def test(net, tests):
    """
    Test the network and return some performances index. Note that the classes must start from 0
    :param tests:   the observations that are going to be used to test the performances of the network
    """

    perf = 0
    for x, y in tests:
        x = np.reshape(x, (x.size, 1))
        y = np.reshape(y, (y.size, 1))

        res = net.feedforward(x)[1][-1]
        if np.argmax(res) == np.argmax(y): perf += 1

    print("{} correctly classified observations ({}%)".format(perf, 100 * perf / len(tests)))
