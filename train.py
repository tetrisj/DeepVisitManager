import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys

states = ('open', 'moved', 'closed')
n_desc = 12
max_t = 20

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(max_t, n_desc)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(states)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def main():
    X = np.ones((1, max_t, n_desc))
    print X
    print model.predict(X, 128)


if __name__ == '__main__':
    main()