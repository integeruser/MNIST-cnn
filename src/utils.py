from __future__ import print_function
import os
import struct
import sys

import numpy as np


def build_mnist_npz(mnist_dirpath):
    training_set_images = os.path.join(mnist_dirpath, "train-images-idx3-ubyte")
    training_set_labels = os.path.join(mnist_dirpath, "train-labels-idx1-ubyte")
    test_set_images = os.path.join(mnist_dirpath, "t10k-images-idx3-ubyte")
    test_set_labels = os.path.join(mnist_dirpath, "t10k-labels-idx1-ubyte")

    with open(training_set_images, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        trn_imgs = np.fromfile(f, dtype=np.uint8).reshape(60000, 1, 28, 28)
    with open(training_set_labels, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        trn_lbls = np.fromfile(f, dtype=np.uint8)

    with open(test_set_images, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        tst_imgs = np.fromfile(f, dtype=np.uint8).reshape(10000, 1, 28, 28)
    with open(test_set_labels, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        tst_lbls = np.fromfile(f, dtype=np.uint8)

    with open("mnist.npz", "wb") as f:
        np.savez_compressed(f, trn_imgs=trn_imgs, trn_lbls=trn_lbls, tst_imgs=tst_imgs, tst_lbls=tst_lbls)

def load_mnist_npz(mnist_npzpath):
    def to_categorical(lbl):
        if lbl not in labels_to_categorical:
            y = np.zeros((10, 1), dtype=np.uint8)
            y[lbl] = 1
            labels_to_categorical[lbl] = y
        return labels_to_categorical[lbl]
    labels_to_categorical = dict()

    dataset = np.load(mnist_npzpath)

    trn_imgs = dataset["trn_imgs"]
    trn_lbls = dataset["trn_lbls"]
    trn_x = np.array([img/255. for img in trn_imgs])
    trn_y = np.array([to_categorical(lbl) for lbl in trn_lbls]).astype(np.uint8)

    tst_imgs = dataset["tst_imgs"]
    tst_lbls = dataset["tst_lbls"]
    tst_x = np.array([img/255. for img in tst_imgs]).astype(np.float32)
    tst_y = np.array([to_categorical(lbl) for lbl in tst_lbls]).astype(np.uint8)
    return (trn_x, trn_y), (tst_x, tst_y)


class bcolors:
    HEADER    = "\033[95m"
    OKBLUE    = "\033[94m"
    OKGREEN   = "\033[92m"
    WARNING   = "\033[93m"
    FAIL      = "\033[91m"
    ENDC      = "\033[0m"
    BOLD      = "\033[1m"
    UNDERLINE = "\033[4m"

def bar(now, end):
    return "[%-10s]" % ("=" * int(10*now/end))

def print(s="", bcolor=None, override=False):
    if print.last_override and not override:
        sys.stdout.write("\n")
    print.last_override = override

    if override:
        sys.stdout.write("\33[2K\r")
    if bcolor:
        sys.stdout.write(bcolor)
    sys.stdout.write(str(s))
    if bcolor:
        sys.stdout.write(bcolors.ENDC)
    if not override and s:
        sys.stdout.write("\n")
    sys.stdout.flush()

print.last_override = False
