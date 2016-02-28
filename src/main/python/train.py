import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np

# time_delta, target_domain_id, target_path_id, referer_domain_id, referer_path_id, prev_domain_id, prev_path_id
from input import event_data
from util import sliding_window
import numpy as np



#states = ('open', 'moved', 'not-mine', 'closed')
states = ('open', 'moved')
n_desc = 14725
max_t = 5


model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(max_t, n_desc)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(states)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


### Feature structure
# timestamp[float], request_id[float], referer_id[float], prev_id[float]

def main():
    # Need to run  sudo ldconfig /usr/local/cuda-7.5/lib64 before for some reason
    for x, Y in event_data('/home/jenia/Deep/labeledJson'):
        X = sliding_window(np.pad(x.toarray(), ((max_t - 1, 0), (0, 0)), 'constant', constant_values=-1)
                           , (5, n_desc), (1, 0))
        print X[0, :]
        print x.shape
        print X.shape
        print Y.shape
        print '----'
        print model.predict(X)
        model.fit(X, Y)



if __name__ == '__main__':
    main()
