__author__ = 'F. Cagnin and A. Torcinovich'

import numpy as np

import layers
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
        self.weights:       list containing the weights of each layer, according to their nature (for polling layers NaN
                            is stored instead).
        self.biases:        list containing the biases of each layer, according to their nature (for polling layers NaN
                            is stored instead).
        self.der_cost_func: the derivative of the cost function used for the backpropagation.
        self.zs:            the zetas computed for each layer.
        self.acts:          the activations computed for each layer.
    """

    def __init__(self, layers_info, cost_func_str):
        """Initialize the data. Initial weights and biases are chosen randomly, according to a gaussian distribution.
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
        :param cost_func_str:  a string with the name of the cost functions applied to the net.

        - Usually polling layers are chained to convolutional layers, but in our implementation each layer can be
          chained to any layers, without restriction (though the correctness of the results can't be assured).

        - Possible activation function values: 'sigmoid', 'softmax', 'rect_lin'
        - Possible polling function values: 'mean', 'max', 'lp_norm'
        - Possible cost function values: 'quadratic', 'cross_entropy', 'log_likelihood'
        """

        # initialize the structures
        self.layers = []
        self.weights = []
        self.biases = []
        self.der_cost_func = getattr(functions, 'der_' + cost_func_str)
        # layer_n_el stores the number of neurons of the previous layer, layer_size stores the side dimension of the
        # previous layer (meaningful for CLs and PLs)
        layer_n_el = layers_info[0]
        layer_size = int(layer_n_el ** 0.5)
        # this for cycle starts from the second layer (the input layer is given by the user)
        for layer_info in layers_info[1:]:
            # check for the type of layer
            if layer_info[0] == 'cl':
                # it's a convolutional layer
                kernel_size, stride_length, act_func_str = layer_info[1], layer_info[2], layer_info[3]
                layer_size = (layer_size - kernel_size) // stride_length + 1
                layer = layers.ConvolutionalLayer(layer_size, kernel_size, stride_length, act_func_str)
                self.layers.append(layer)
                self.weights.append(np.reshape(np.random.normal(0, 1 / kernel_size, (kernel_size, kernel_size)),
                                                   (kernel_size, kernel_size)))
                # in any case there is one shared bias, no matter how many layers are in the block
                self.biases.append(np.random.normal(0, 1 / kernel_size, 1))
            elif layer_info[0] == 'pl':
                # it's a polling layer
                kernel_size, poll_func_str, add_params = layer_info[1], layer_info[2], layer_info[3]
                layer_size = layer_size // kernel_size
                layer = layers.PollingLayer(layer_size, kernel_size, poll_func_str, add_params)
                self.layers.append(layer)
                # polling layers haven't associated weights and biases. NaN is stored instead
                self.weights.append(np.NaN)
                self.biases.append(np.NaN)
            elif layer_info[0] == 'fcl':
                # it's a fully connected layer
                layer_size, act_func_str = layer_info[1], layer_info[2]
                layer = layers.FullyConnectedLayer(layer_size, act_func_str)
                self.layers.append(layer)
                self.weights.append(np.reshape(np.random.normal(0, 1 / layer_n_el ** 0.5, (layer_size, layer_n_el)),
                                               (layer_size, layer_n_el)))
                self.biases.append(
                    np.reshape(np.random.normal(0, 1 / layer_n_el ** 0.5, layer_size), (layer_size, 1)))
            layer_size = layer.size
            layer_n_el = layer.num_neurons

    def feedforward(self, x):
        """Perform the feedforward of one observation and return the list of
        z and activations of each layer
        :param x: the observation taken as input layer
        :return self.zs:    the list of zetas of each layer
                self.acts:  the list of activations each layer
        """

        # the input layer hasn't zetas and its initial values are considered directly as activations
        self.zs = [np.NaN]
        self.acts = [x]

        # feedforward the input for each layer
        for layer, w, b in zip(self.layers, self.weights, self.biases):
            z, a = layer.feedforward(self.acts[-1], w, b)
            self.zs.append(z)
            self.acts.append(a)

        return self.zs, self.acts

    def backpropagate(self, y):
        """Perform the backpropagation for one observation and return the derivative of the weights relative to each
        layer
        :param y:  the class of the current observation represented in 1-of-k coding

        :return d_der_ws:   list containing the amount of change in the weights, due to the current observation
                d_der_bs:   list containing the amount of change in the biases, due to the current observation
        """

        # initialize the structures
        d_der_ws = []
        d_der_bs = []
        delta_zlp = self.der_cost_func(self.acts[-1], y) * self.layers[-1].der_act_func(self.zs[-1])
        for layer, z, a, w in zip(reversed(self.layers), reversed(self.zs[:-1]), reversed(self.acts[:-1]),
                                  reversed(self.weights)):
            d_der_w, d_der_b, delta_zlp = layer.backpropagate(z, a, w, delta_zlp)
            d_der_ws.insert(0, d_der_w)
            d_der_bs.insert(0, d_der_b)
        return d_der_ws, d_der_bs


def train(net, inputs, epochs, batch_size, eta):
    """Train the network according to the Stochastic Gradient Descent (SGD)
     algorithm
    :param inputs:     the observations that are going to be used to train the network
    :param epochs:     the number of epochs
    :param batch_size: the size of the batch used in single cycle of the SGD
    :param eta:        the learning rate
    """

    # check the correctness of the input
    if eta <= 0:
        raise ValueError("eta is not positive")

    n_obs = len(inputs)
    if n_obs % batch_size:
        raise ValueError("the number of observations is not a multiple of 'batch_size'")

    # initialize y
    y = np.zeros((net.layers[-1].size, 1))
    # for each epoch...
    for i in range(0, epochs):
        print("Epoch ", i + 1)
        # shuffle the observations
        np.random.shuffle(inputs)
        # for each batch...
        for j in range(0, n_obs, batch_size):
            # initialize the structures
            der_weights = [np.zeros(w.shape) if not isinstance(w, float) else np.NaN for w in net.weights]
            der_biases = [np.zeros(b.shape) if not isinstance(b, float) else np.NaN for b in net.biases]
            # for each observation in the current batch...
            for x, lab in inputs[j:j + batch_size]:
                # generate the 1-of-k coding of the current observation class
                y[lab] = 1
                # feedforward the observation
                net.feedforward(x)
                # backpropagate the error
                (d_der_ws, d_der_bs) = net.backpropagate(y)
                # sum the derivatives of the weights and biases of the current observation to the previous ones
                der_weights = [dw + ddw for dw, ddw in zip(der_weights, d_der_ws)]
                der_biases = [db + ddb for db, ddb in zip(der_biases, d_der_bs)]
                # reset y
                y[lab] = 0
            # update weights and biases with the results of the current batch
            net.weights = [w - eta / batch_size * dw for w, dw in zip(net.weights, der_weights)]
            net.biases = [b - eta / batch_size * db for b, db in zip(net.biases, der_biases)]


def test(net, tests):
    """Test the network and return some performances index. Note that the classes must start from 0.
    :param tests:  the observations that are going to be used to test the performances of the network
    """

    perf = 0
    # for each observation...
    for x, lab in tests:
        # retrieve the index of the output neuron with the maximum activation
        res = np.argmax(net.feedforward(x)[1][-1])
        # if it's equal to the label increment perf
        if lab == res:
            perf += 1
    # display the number of correct observations along with its percentage
    print(perf, " correctly classified observations (", 100 * perf / len(tests), "%)")
