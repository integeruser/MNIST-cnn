#!/usr/bin/env python2
import argparse
import sys

import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD

import utils as u

trn_set, tst_set = u.load_mnist_npz(sys.argv[1])

trn_x, trn_y = trn_set
trn_y = np.squeeze(trn_y, axis=2)

tst_x, tst_y = tst_set
tst_y = np.squeeze(tst_y, axis=2)

################################################################################

model = Sequential()

model.add(Convolution2D(2, 5, 5, activation='sigmoid', input_shape=(1, 28, 28),
                        weights=[np.zeros((2, 1, 5, 5)), np.zeros(2)]))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(10, activation='softmax',
                weights=[np.zeros((128, 10)), np.zeros(10)]))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))

model.fit(trn_x, trn_y, nb_epoch=1, batch_size=10)
score = model.evaluate(tst_x, tst_y, show_accuracy=True)
print 'Test score:', score[0]
print 'Test accuracy:', score[1]
