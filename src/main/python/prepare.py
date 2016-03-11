import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from pyspark import SparkContext, SQLContext, SparkConf
from pyspark.storagelevel import StorageLevel
from itertools import islice
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

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def train_chunk(X, Y, model):
    try:
        model.fit(X, Y, nb_epoch=2, batch_size=1024, show_accuracy=True)
        model.fit_generator()
        model.save_weights('/home/jenia/Deep/visit.h5', overwrite=True)
    except:
        import traceback
        traceback.print_exc()
    finally:
        X = None
        Y = None
    return X, Y


def open_sparse(x):
    return np.array([m.toarray() for m in x])


def sample_iterator(combined_df):
    ''' Repatedly sample from dataframe then iterate over partitions, finally yeilding fixed size batches
    '''
    sample_fraction = 0.1  # Mostly used to keep some randomness while still retaining big enough batches
    batch_size = 4096
    X_batch = None
    Y_batch = None
    while True:
        training_rdd = combined_df.sample(fraction=sample_fraction, withReplacement=True).map(learning_set)
        for x, y in training_rdd.toLocalIterator():
            X = open_sparse(x)
            if X_batch is None:
                X_batch = X
                Y_batch = y
            else:
                X_batch = np.vstack([X_batch, X])
                Y_batch = np.vstack([Y_batch, y])
            if X_batch.shape[0] > batch_size:
                yield X_batch, Y_batch
                X_batch = None
                Y_batch = None


def main():
    conf = SparkConf().set("spark.sql.shuffle.partitions", "4096") \
        .set("spark.python.worker.memory", "2g") \
        .set("spark.memory.storageFraction", "0.1")

    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    combined_df = sqlContext.read.parquet('/home/jenia/Deep/combined_train/')
    samples_to_train = 512 * 256  # Should be big enough to generalize
    sample_fraction = 0.1
    save_callback = ModelCheckpoint(filepath='/home/jenia/Deep/visit.h5')
    #    combinedDF = sqlContext.read.parquet(
    #        '/home/jenia/Deep/combined/part-r-00000-fea4cef7-4875-49b0-a9db-4904a007a852.gz.parquet')

    model = make_model()
    model.load_weights('/home/jenia/Deep/visit.h5')

    test_combined_df = sqlContext.read.parquet('/home/jenia/Deep/combined_test/')
    validation_data = sample_iterator(test_combined_df).next()

    for i in xrange(100):
        training_rdd = combined_df.sample(fraction=sample_fraction, withReplacement=True).map(learning_set)
        training_generator = sample_iterator(combined_df)
        model.fit_generator(training_generator,
                            samples_per_epoch=samples_to_train,
                            nb_epoch=200,
                            show_accuracy=True,
                            nb_worker=2,
                            callbacks=[save_callback],
                            validation_data=validation_data)


if __name__ == '__main__':
    main()
