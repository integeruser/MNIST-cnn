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

        self.loss_func = getattr(f, loss_func)

        self.weights = dict()
        self.biases = dict()
        for prev_layer, layer in self.layers:
            layer.connect_to(prev_layer)

            prev_layer_nout = prev_layer.depth * prev_layer.height * prev_layer.width
            layer_nin       = prev_layer_nout
            layer_nout      = layer.depth      * layer.height      * layer.width
            if type(layer) is l.FullyConnectedLayer:
                w_shape = (layer_nout, prev_layer_nout)
                b_shape = (layer_nout, 1)
            elif type(layer) is l.ConvolutionalLayer:
                w_shape = (layer.depth, prev_layer.depth, layer.kernel_size, layer.kernel_size)
                b_shape = (layer.depth, 1)
            elif type(layer) is l.PollingLayer:
                w_shape = (0)
                b_shape = (0)
            else:
                raise NotImplementedError
            self.weights[layer] = u.glorot_uniform(w_shape, layer_nin, layer_nout).astype(np.float32)
            self.biases[layer]  = np.zeros(b_shape).astype(np.float32)

    def __str__(self):
        s = "NeuralNetwork([\n"
        for prev_layer, layer in self.layers:
            s += "    %s,\n" % prev_layer
        s += "    %s\n" % self.output_layer
        s += "], loss_func=%s)" % self.loss_func.__name__
        return s

    def feedforward(self, x):
        self.input_layer.z = x
        self.input_layer.a = x

        for prev_layer, layer in self.layers:
            w = self.weights[layer]
            b = self.biases[layer]
            layer.feedforward(prev_layer, w, b)

    def backpropagate(self, batch, optimizer):
        der_weights = {layer: np.zeros_like(self.weights[layer]) for _, layer in self.layers}
        der_biases  = {layer: np.zeros_like(self.biases[layer])  for _, layer in self.layers}

        for x, y in batch:
            self.feedforward(x)

            # propagate the error backward
            delta = self.loss_func(self.output_layer.a, y) * self.output_layer.der_act_func(self.output_layer.z, y)
            for prev_layer, layer in reversed(self.layers):
                w = self.weights[layer]
                der_w, der_b, delta = layer.backpropagate(prev_layer, w, delta)
                der_weights[layer] += der_w
                der_biases[layer]  += der_b

        # update weights and biases
        if optimizer == "adadelta":
            ro  = 0.95
            eps = 1e-8
            gsum_weights = {layer: 0 for _, layer in self.layers}
            xsum_weights = {layer: 0 for _, layer in self.layers}
            gsum_biases  = {layer: 0 for _, layer in self.layers}
            xsum_biases  = {layer: 0 for _, layer in self.layers}
        for _, layer in self.layers:
            gw = der_weights[layer]/len(batch)
            gb = der_biases[layer] /len(batch)

            if optimizer["type"] == "SGD":
                self.weights[layer] += -optimizer["eta"]*gw
                self.biases[layer]  += -optimizer["eta"]*gb
            elif optimizer["type"] == "adadelta":
                gsum_weights[layer] = ro*gsum_weights[layer] + (1-ro)*gw*gw
                dx = -np.sqrt((xsum_weights[layer]+eps)/(gsum_weights[layer]+eps)) * gw
                self.weights[layer] += dx
                xsum_weights[layer] = ro*xsum_weights[layer] + (1-ro)*dx*dx

                gsum_biases[layer]  = ro*gsum_biases[layer]  + (1-ro)*gb*gb
                dx = -np.sqrt((xsum_biases[layer] +eps)/(gsum_biases[layer] +eps)) * gb
                self.biases[layer]  += dx
                xsum_biases[layer]  = ro*xsum_biases[layer]  + (1-ro)*dx*dx
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
        for j, batch in enumerate(batches):
            net.backpropagate(batch, optimizer)
            u.print("Epoch %02d %s [%d/%d]" % (i+1, u.bar(j+1, len(batches)), j+1, len(batches)), override=True)

        if vld_set:
            # test the net at the end of each epoch
            u.print("Epoch %02d %s [%d/%d] > Testing..." % (i+1, u.bar(j+1, len(batches)), j+1, len(batches)), override=True)
            accuracy = test(net, vld_set)
            u.print("Epoch %02d %s [%d/%d] > Accuracy: %0.2f%%" % (i+1, u.bar(j+1, len(batches)), j+1, len(batches), accuracy), override=True)
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
    accuracy *= 100/len(tests)
    return accuracy
