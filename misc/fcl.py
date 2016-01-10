#!/usr/bin/env python2
import argparse
import sys

import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

import utils as u

trn_set, tst_set = u.load_mnist_npz(sys.argv[1])

trn_x, trn_y = trn_set
trn_x = np.squeeze(trn_x, axis=1).reshape((60000, 784))
trn_y = np.squeeze(trn_y, axis=2)

tst_x, tst_y = tst_set
tst_x = np.squeeze(tst_x, axis=1).reshape((10000, 784))
tst_y = np.squeeze(tst_y, axis=2)

################################################################################

model = Sequential()

model.add(Dense(10, activation='softmax', input_dim=28*28,
                weights=[np.zeros((28*28, 10)), np.zeros(10)]))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))

model.fit(trn_x, trn_y, nb_epoch=1, batch_size=10)
score = model.evaluate(tst_x, tst_y, show_accuracy=True)
print 'Test score:', score[0]
print 'Test accuracy:', score[1]
