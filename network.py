import numpy as np

from layers import FullyConnectedLayer
import functions


class NeuralNetwork():
    def __init__(self, input_layer, layers, cost_func):
        """
        Initialize the data. Initial input_weights and input_biases are chosen randomly from a gaussian distribution.

        :param layers: a vector which stores each layer of the NN.
        :param cost_func: the cost function applied to the net
        """
        self.input_layer = input_layer
        self.layers = layers
        self.der_cost_func = functions.get_derivative(cost_func)

        self.input_weights = {}
        self.input_biases = {}
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

    def feedforward(self, x):
        """
        Perform the feedforward of one observation and return the list of z and activations of each layer

        :param x: the observation taken as input layer
        :returns: the zetas and the activations of each layer
        """
        # the input layer hasn't zetas and its initial values are considered directly as activations
        zs = {self.input_layer: np.array([])}
        acts = {self.input_layer: x}

        # feedforward the input for each layer
        prev_layer = self.input_layer
        for layer in self.layers:
            w = self.input_weights[layer]
            b = self.input_biases[layer]
            zs[layer], acts[layer] = layer.feedforward(acts[prev_layer], w, b)
            prev_layer = layer
        return zs, acts

    def backpropagate(self, batch, eta):
        """
        Perform the backpropagation for one observation and return the derivative of the input_weights relative to each layer

        :param y: the class of the current observation represented in 1-of-k coding
        """
        der_weights = {layer: np.zeros(self.input_weights[layer].shape) for layer in self.layers}
        der_biases = {layer: np.zeros(self.input_biases[layer].shape) for layer in self.layers}

        # for each observation in the current batch
        for x, y in batch:
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

    :param net: the network to train
    :param inputs: the observations that are going to be used to train the network
    :param num_epochs: the number of epochs
    :param batch_size: the size of the batch used in single cycle of the SGD
    :param eta: the learning rate
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

    :param tests: the observations that are going to be used to test the performances of the network
    """
    perf = 0
    for x, y in tests:
        res = net.feedforward(x)[1][net.layers[-1]]
        if np.argmax(res) == np.argmax(y): perf += 1

    print("{} correctly classified observations ({}%)".format(perf, 100 * perf / len(tests)))
