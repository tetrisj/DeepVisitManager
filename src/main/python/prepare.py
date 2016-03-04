import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from pyspark import SparkContext, SQLContext
from scipy import sparse

# Move to global config
n_desc = 1669
max_t = 5
# states = ('open', 'moved', 'not-mine', 'closed')
states = ('open', 'moved')


def event_features(event, start_time):
    feature_dict = {(0, int(k)): v for k, v in event.feature.items()}
    feature_dict[(0, n_desc - 1)] -= start_time  # Adjust timestamp feature to be relative
    features = sparse.dok_matrix((1, n_desc))
    features.update(feature_dict)
    return features.tocsr()


def learning_set(combined):
    ''' Creates data of shape n_samples, max_t, n_desc and labels of shape n_smaples
        Each sample contains all valid events in the visit up to that point and then the tested event
        The label is the decision for that combination of events
        The idea is to simulate a state machine that decides whether a new event belongs to the visit
    '''
    timestamps = np.array(combined.timestamps) / 1000.0
    start_time = timestamps.min()
    event_timestamps = np.array(
            [event.event.timestamp for event in combined.events if event.event.timestamp > start_time - 0.01])

    # Match events to visit timestamps

    intersect = np.in1d(event_timestamps, timestamps) * 1  # Events with matching timestamps get a 1 label
    true_indices = intersect.nonzero()[0]
    n_samples = len(intersect)

    # Calculate labels (dimension per label)
    Y = np.array([intersect, 1 - intersect]).T

    event_data = sparse.vstack([event_features(event, start_time) for event in combined.events if
                                event.event.timestamp > start_time - 0.01])

    X = [sparse.csr_matrix((max_t, n_desc)) for i in range(n_samples)]
    # TODO: use something more efficient than iterations
    for i in range(n_samples):
        X[i][max_t - 1] = event_data[i]
        true_for_i = [idx for idx in true_indices if idx < i][-max_t + 1:]
        if true_for_i:
            X[i][max_t - 1 - len(true_for_i): max_t - 1] = event_data[true_for_i]

    return X, Y


def make_model():
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(max_t, n_desc)))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(states)))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def main():
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    #    combinedDF = sqlContext.read.parquet('/home/jenia/Deep/combined/')
    combinedDF = sqlContext.read.parquet(
        '/home/jenia/Deep/combined/')

    model = make_model()
    training = combinedDF.rdd.coalesce(2048, True).map(learning_set).toLocalIterator()

    samples_to_train = 128 * 64
    X = None
    Y = None
    for x, y in training:
        if X is None:
            X = np.array([m.toarray() for m in x])
            Y = y
        else:
            X = np.vstack([X, np.array([m.toarray() for m in x])])
            Y = np.vstack([Y, y])

        if X.shape[0] >= samples_to_train:
            try:
                model.fit(X, Y, nb_epoch=32, batch_size=128)
                model.save_weights('/home/jenia/Deep/visit.h5', overwrite=True)
            except:
                import traceback
                traceback.print_exc()
            finally:
                X = None
                Y = None


if __name__ == '__main__':
    main()
