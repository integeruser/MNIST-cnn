# kanji-ai
This repository contains our project for the Artificial Intelligence course.

## Usage
To start, download the well-known MNIST data set (four `.gz` archives) from the the [official location](http://yann.lecun.com/exdb/mnist/) and decompress it:
```
Downloads ➤ ls
t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz  train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz
Downloads ➤ gzip -d *
Downloads ➤ ls
t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte  train-images-idx3-ubyte train-labels-idx1-ubyte
```
Then, use `build_mnist_npz()` from `utils.py` to build `mnist.npz`:
```
kanji-ai ➤ python3
Python 3.5.0 (default, Sep 23 2015, 04:41:38)
[GCC 4.2.1 Compatible Apple LLVM 7.0.0 (clang-700.0.72)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import utils
>>> utils.build_mnist_npz('/Users/fcagnin/Downloads')
>>>
kanji-ai ➤ file mnist.npz
mnist.npz: Zip archive data, at least v2.0 to extract
```

Finally, run the included examples:
```
kanji-ai ➤ time python3 -B mnist_fcl.py mnist.npz
Loading 'mnist.npz'...
Building NN...
Training NN...
Epoch 1
Testing NN...
9574 correctly classified observations (95.74%)
python3 -B mnist_fcl.py mnist.npz  26.40s user 4.30s system 135% cpu 22.577 total

kanji-ai ➤ time python3 -B mnist_cnn.py mnist.npz
Loading 'mnist.npz'...
Building CNN...
Training CNN...
Epoch 1
Testing CNN...
6576 correctly classified observations (65.76%)
python3 -B mnist_cnn.py mnist.npz  548.65s user 1.35s system 99% cpu 9:10.50 total
```

## License
The MIT License (MIT)

Copyright (c) 2015 Alessandro Torcinovich, Francesco Cagnin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
