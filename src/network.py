import numpy as np

import functions as f
import layers as l
import utils as u


class NeuralNetwork():
    def __init__(self, layers, loss_func):
        assert len(layers) > 0

        assert isinstance(layers[0], l.InputLayer)
        self.input_layer = layers[0]

        assert isinstance(layers[-1], l.FullyConnectedLayer)
        self.output_layer = layers[-1]

        self.layers = [(prev_layer, layer) for prev_layer, layer in zip(layers[:-1], layers[1:])]

        self.loss_func = loss_func

        for prev_layer, layer in self.layers:
            layer.connect_to(prev_layer)

    def feedforward(self, x):
        self.input_layer.z = x
        self.input_layer.a = x

        for prev_layer, layer in self.layers:
            layer.feedforward(prev_layer)

    def backpropagate(self, batch, optimizer):
        der_weights = {layer: np.zeros_like(layer.w) for _, layer in self.layers}
        der_biases  = {layer: np.zeros_like(layer.b)  for _, layer in self.layers}

        for x, y in batch:
            self.feedforward(x)

            # propagate the error backward
            loss = self.loss_func(self.output_layer.a, y)
            delta = loss * self.output_layer.der_act_func(self.output_layer.z, y)
            for prev_layer, layer in reversed(self.layers):
                der_w, der_b, delta = layer.backpropagate(prev_layer, delta)
                der_weights[layer] += der_w
                der_biases[layer]  += der_b

        # update weights and biases
        if optimizer["type"] == "adadelta":
            rho = 0.95
            eps = 1e-8
            gsum_weights = {layer: 0 for _, layer in self.layers}
            xsum_weights = {layer: 0 for _, layer in self.layers}
            gsum_biases  = {layer: 0 for _, layer in self.layers}
            xsum_biases  = {layer: 0 for _, layer in self.layers}
        for _, layer in self.layers:
            gw = der_weights[layer]/len(batch)
            gb = der_biases[layer] /len(batch)

            if optimizer["type"] == "SGD":
                layer.w += -optimizer["eta"]*gw
                layer.b += -optimizer["eta"]*gb
            elif optimizer["type"] == "adadelta":
                gsum_weights[layer] = rho*gsum_weights[layer] + (1-rho)*gw*gw
                dx = -np.sqrt((xsum_weights[layer]+eps)/(gsum_weights[layer]+eps)) * gw
                layer.w += dx
                xsum_weights[layer] = rho*xsum_weights[layer] + (1-rho)*dx*dx

                gsum_biases[layer]  = rho*gsum_biases[layer]  + (1-rho)*gb*gb
                dx = -np.sqrt((xsum_biases[layer] +eps)/(gsum_biases[layer] +eps)) * gb
                layer.b += dx
                xsum_biases[layer]  = rho*xsum_biases[layer]  + (1-rho)*dx*dx
            else:
                raise NotImplementedError


def train(net, optimizer, num_epochs, batch_size, trn_set, vld_set=None):
    assert isinstance(net, NeuralNetwork)
    assert num_epochs > 0
    assert batch_size > 0

    trn_x, trn_y = trn_set
    inputs = [(x, y) for x, y in zip(trn_x, trn_y)]

    for i in range(num_epochs):
        np.random.shuffle(inputs)

        # divide input observations into batches
        batches = [inputs[j:j+batch_size] for j in range(0, len(inputs), batch_size)]
        inputs_done = 0
        for j, batch in enumerate(batches):
            net.backpropagate(batch, optimizer)
            inputs_done += len(batch)
            u.print("Epoch %02d %s [%d/%d]" % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs)), override=True)

        if vld_set:
            # test the net at the end of each epoch
            u.print("Epoch %02d %s [%d/%d] > Testing..." % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs)), override=True)
            accuracy = test(net, vld_set)
            u.print("Epoch %02d %s [%d/%d] > Accuracy: %0.2f%%" % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs), accuracy*100), override=True)
        u.print()

def test(net, tst_set):
    assert isinstance(net, NeuralNetwork)

    tst_x, tst_y = tst_set
    tests = [(x, y) for x, y in zip(tst_x, tst_y)]

    accuracy = 0
    for x, y in tests:
        net.feedforward(x)
        if np.argmax(net.output_layer.a) == np.argmax(y):
            accuracy += 1
    accuracy /= len(tests)
    return accuracy
