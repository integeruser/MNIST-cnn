# MNIST-cnn
This repository contains a Python 3 naïve implementation of a neural network with convolutional and pooling layers, useful for educational purposes. It was tested with satisfactory results the on the well-known [MNIST](http://yann.lecun.com/exdb/mnist/) data set.

Alessandro and Francesco


## Prerequisites
The code makes heavy use of [NumPy](http://www.numpy.org/). Install it using pip:
```
~ ➤ pip3 install --user numpy
```

Then, download the MNIST data set (four `.gz` archives) and decompress it:
```
~ ➤ cd Downloads
Downloads ➤ ls
t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz  train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz
Downloads ➤ gzip -d *
Downloads ➤ ls
t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte  train-images-idx3-ubyte train-labels-idx1-ubyte
```
**Note**: on Windows, as per our tests, the extracted files will have a different name of the ones showed above (e.g. `train-images.idx3-ubyte` instead of `train-images-idx3-ubyte`). You can either manually rename the extracted files or modify lines 10-13 of `src/utils.py`.


## Usage
Convert the downloaded data set to the NPZ binary data format using the function `build_mnist_npz()` from `src/utils.py`:
```
MNIST-cnn ➤ cd src
src ➤ ls
examples.py  functions.py layers.py    network.py   utils.py
src ➤ python3 -q
>>> import utils
>>> utils.build_mnist_npz('/Users/fcagnin/Downloads')
>>> exit()
src ➤ file mnist.npz
mnist.npz: Zip archive data, at least v2.0 to extract
```

Finally, run any of the included examples:
```
src ➤ time python3 -OO examples.py mnist.npz fcl01
Loading 'mnist.npz'...
Loading 'fcl01'...
def fcl01():
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.FullyConnectedLayer(100, init_func=f.glorot_uniform, act_func=f.sigmoid),
        l.FullyConnectedLayer(10, init_func=f.glorot_uniform, act_func=f.sigmoid)
    ], f.quadratic)
    optimizer = o.SGD(3.0)
    num_epochs = 1
    batch_size = 10
    return net, optimizer, num_epochs, batch_size
Training network...
Epoch 01 [==========] [50000/50000] > Validation accuracy: 95.41%
Testing network...
Test accuracy: 95.07%
python3 -OO examples.py mnist.npz fcl01  24.10s user 4.07s system 130% cpu 21.517 total
```
```
src ➤ time python3 examples.py mnist.npz cnn01
Loading 'mnist.npz'...
Loading 'cnn01'...
def cnn01():
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.ConvolutionalLayer(2, kernel_size=5, init_func=f.glorot_uniform, act_func=f.sigmoid),
        l.MaxPoolingLayer(pool_size=2),
        l.FullyConnectedLayer(height=10, init_func=f.glorot_uniform, act_func=f.softmax)
    ], f.log_likelihood)
    optimizer = o.SGD(0.1)
    num_epochs = 3
    batch_size = 10
    return net, optimizer, num_epochs, batch_size
Training network...
Epoch 01 [==========] [50000/50000] > Validation accuracy: 88.86%
Epoch 02 [==========] [50000/50000] > Validation accuracy: 89.84%
Epoch 03 [==========] [50000/50000] > Validation accuracy: 89.11%
Testing network...
Test accuracy: 88.53%
python3 examples.py mnist.npz cnn01  2869.89s user 11.16s system 99% cpu 45:12.89 total
```
