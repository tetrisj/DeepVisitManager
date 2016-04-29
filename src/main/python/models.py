from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop


def make_multi_model(max_t=32):
    from input import max_visits_count, n_desc
    model = Sequential()
    model.add(LSTM(256, return_sequences=True,
                   input_shape=(max_t, n_desc)))
    model.add(LSTM(128, return_sequences=False,
                   input_shape=(max_t, n_desc)))
    model.add(Dense(max_visits_count))
    model.add(Activation('softmax'))

    rmsprop = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=["accuracy"])
    # sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer=sgd)
    return model


def make_visit_model(max_t=5):
    from input import max_visits_count, n_desc
    model = Sequential()
    model.add(LSTM(768, return_sequences=True,
                   input_shape=(max_t, n_desc)))
    model.add(LSTM(256, return_sequences=False,
                   input_shape=(max_t, n_desc)))
    model.add(Dense(max_visits_count))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
    # sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer=sgd)
    return model


def make_model(mode):
    if mode == 'visit':
        return make_visit_model()
    else:
        return make_multi_model()
