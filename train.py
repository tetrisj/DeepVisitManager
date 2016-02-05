import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np

# time_delta, target_domain_id, target_path_id, referer_domain_id, referer_path_id, prev_domain_id, prev_path_id

states = ('open', 'moved', 'not-mine', 'closed')
n_desc = 7
max_t = 20

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(max_t, n_desc)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(states)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

### Feature structure



def main():
    X = np.ones((1, max_t, n_desc))
    Y = np.ones((1, 1))
    print model.predict(X, 128)
    model.fit(X, Y)
    print model.predict(X, 128)


if __name__ == '__main__':
    main()
