import numpy as np
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback
from pyspark import SparkContext, SQLContext, SparkConf
from keras.regularizers import l2
from scipy import sparse
from sklearn.metrics import confusion_matrix
import os
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

import warnings

warnings.filterwarnings("ignore")
# Move to global config
n_desc = 3 * 201 + 2
max_visits_count = 16
max_t = 32
states = ('open', 'moved')
start_vector = np.ones((max_t - 1, n_desc)) * -1  # Indicates start of sequence (since we are using stateful RNNs)
batch_size = 128


# states = ('open', 'moved')


def event_features(event, start_time):
    feature = np.array([event.feature.requestVec, event.feature.hrefVec, event.feature.prevVec]).reshape(
        (1, n_desc - 2))
    return np.append(feature, event.event.timestamp - start_time)


def learning_set_visits(combined):
    ''' Creates data of shape n_samples, max_t, n_desc and labels of shape n_samples
        Each sample contains all valid events in the visit up to that point and then the tested event
        The label is the decision for that combination of events
        The idea is to simulate a state machine that decides whether a new event belongs to the visit
    '''

    events = sorted(combined.events, key=lambda e: e.event.timestamp)
    visits = sorted(combined.visits, key=lambda v: v.timestamps[0])
    if len(visits) > max_visits_count:
        return None, None
    n_samples = len(events)
    event_timestamps = np.array([event.event.timestamp for event in events])
    start_time = event_timestamps.min()
    labels = np.zeros(n_samples)
    # Calculate labels
    for i, visit in enumerate(visits):
        intersect = np.in1d(event_timestamps,
                            np.array(visit.timestamps))  # Events with matching timestamps get a 1 label
        labels[intersect.nonzero()[0]] = i + 1

    # Remove events not caught by any visit
    good_labels = labels[labels.nonzero()]
    # If there is a visit without matching events - skip
    if not np.array_equal(np.unique(good_labels), np.array(range(1, len(combined.visits) + 1))):
        return None, None

    X = np.vstack([event_features(event, start_time) for event in events])
    X = X[labels.nonzero()]

    # Add labels to all features (the last one will be removed later after we make windows)
    X = np.hstack([X, good_labels.reshape((good_labels.shape[0], 1))])
    X = np.vstack([start_vector, X])

    # Make train predictions
    Y = to_categorical(good_labels - 1, max_visits_count)  # indicator matrix with labels starting at 0

    # Create learning set windows
    X = X.reshape((1, X.shape[0], n_desc))
    X = np.vstack([X[:, i:i + max_t] for i in range(X.shape[1] - max_t + 1)])

    # Remove last label (because this is what we want to predict)
    X[:, -1, -1] = -1

    return X, Y


def learning_set(combined):
    ''' Creates data of shape n_samples, max_t, n_desc and labels of shape n_samples
        Each sample contains all valid events in the visit up to that point and then the tested event
        The label is the decision for that combination of events
        The idea is to simulate a state machine that decides whether a new event belongs to the visit
    '''

    events = sorted(combined.events, key=lambda e: e.event.timestamp)
    visits = sorted(combined.visits, key=lambda v: v.timestamps[0])
    if len(visits) > max_visits_count:
        return None, None
    n_samples = len(events)
    event_timestamps = np.array([event.event.timestamp for event in events])
    start_time = event_timestamps.min()
    labels = np.zeros(n_samples)
    # Calculate labels
    for i, visit in enumerate(visits):
        intersect = np.in1d(event_timestamps,
                            np.array(visit.timestamps))  # Events with matching timestamps get a 1 label
        labels[intersect.nonzero()[0]] = i + 1

    # Remove events not caught by any visit
    good_labels = labels[labels.nonzero()]
    # If there is a visit without matching events - skip
    if not np.array_equal(np.unique(good_labels), np.array(range(1, len(combined.visits) + 1))):
        return None, None

    X = np.vstack([event_features(event, start_time) for event in events])
    X = X[labels.nonzero()]

    # Add labels to all features (the last one will be removed later after we make windows)
    X = np.hstack([X, good_labels.reshape((good_labels.shape[0], 1))])
    X = np.vstack([start_vector, X])

    # Find visit transfers. 0 means direct and is default and the last visit can't be sending
    sending_map = np.zeros(len(visits))
    for i, visit in enumerate(visits):
        if not visit.sendingPage:
            continue
        landing_time = visit.timestamps[0]
        for j, other_visit in enumerate(visits):
            if j >= i:
                continue
            candidate_pages = (p for t, p in zip(other_visit.timestamps, other_visit.pages) if t < landing_time)
            if visit.sendingPage in candidate_pages:
                sending_map[i] = j + 1

    # Make train predictions
    Y = to_categorical(sending_map[(good_labels - 1).astype(int)],
                       max_visits_count)  # indicator matrix with labels starting at 0

    # Create learning set windows
    X = X.reshape((1, X.shape[0], n_desc))
    X = np.vstack([X[:, i:i + max_t] for i in range(X.shape[1] - max_t + 1)])

    # Remove last label (because this is what we want to predict)
    X[:, -1, -1] = -1

    return X, Y


def make_model():
    model = Sequential()
    model.add(LSTM(256, return_sequences=True,
                   input_shape=(max_t, n_desc)))
    model.add(LSTM(128, return_sequences=False,
                   input_shape=(max_t, n_desc)))
    model.add(Dense(max_visits_count))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def concurrent_local_iterator(rdd, concurrency=32):
    partition_count = rdd.getNumPartitions()
    for start_partition in range(0, partition_count, concurrency):
        end_partition = min(partition_count, start_partition + concurrency)
        partitions = range(start_partition, end_partition)
        rows = rdd.context.runJob(rdd, lambda x: x, partitions)
        for row in rows:
            yield row


def sample_iterator(df, sample_fraction=0.5, batch_size=3 * 1024):
    ''' Repatedly sample from dataframe then iterate over partitions, finally yeilding fixed size batches
    :param df: dataframe to sample
    :param batch_size: Number of samples in each yielded batch
    '''
    X = None
    Y = None
    while True:
        samples_rdd = df.sample(fraction=sample_fraction, withReplacement=True).map(learning_set)
        samples_iterator = concurrent_local_iterator(samples_rdd)
        for x, y in samples_iterator:
            if x is None:
                continue
            if X is None:
                X = x
                Y = y
            else:
                X = np.vstack([X, x])
                Y = np.vstack([Y, y])
            if X.shape[0] > batch_size:
                yield X, Y
                X = None
                Y = None


class WriteLosses(Callback):
    def __init__(self, filepath):
        super(WriteLosses, self).__init__()
        self.path = filepath

    def on_train_begin(self, logs={}):
        self.f = file(self.path, 'wb')
        self.f.write('type,loss,accuracy\n')

    def on_batch_end(self, batch, logs={}):
        self.f.write('batch,%s,%s\n' % (logs.get('loss'), logs.get('acc')))

    def on_epoch_end(self, batch, logs={}):
        self.f.write('epoch,%s,%s\n' % (logs.get('val_loss'), logs.get('val_acc')))
        self.f.flush()

    def on_train_end(self, logs={}):
        self.f.close()


class ConfusionMatrix(Callback):
    def __init__(self, validation_data):
        super(ConfusionMatrix, self).__init__()
        self.validation_data = validation_data

    def print_confusion_matrix(self):
        res = np.argmax(self.model.predict(self.validation_data[0]), 1)
        labels = np.argmax(self.validation_data[1], 1)
        print '\nConfusion matrix:'
        print confusion_matrix(res, labels)

    def on_epoch_begin(self, batch, logs={}):
        self.print_confusion_matrix()


def main():
    conf = SparkConf().set("spark.sql.shuffle.partitions", "4096") \
        .set("spark.python.worker.memory", "2g") \
        .set("spark.memory.storageFraction", "0.1") \
        .set("spark.ui.showConsoleProgress", "false")

    model_save_path = '/home/jenia/Deep/transfer.h5'
    log_path = '/home/jenia/Deep/train.log'
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    combined_df = sqlContext.read.parquet('/home/jenia/Deep/combined_train/')
    samples_per_epoch = 1024 * 128

    save_callback = ModelCheckpoint(filepath=model_save_path, save_best_only=False)
    log_callback = WriteLosses(filepath=log_path)

    model = make_model()
    if os.path.isfile(model_save_path):
        print 'Loading model from: %s' % model_save_path
        model.load_weights(model_save_path)

    test_combined_df = sqlContext.read.parquet('/home/jenia/Deep/combined_test/')
    validation_data = sample_iterator(test_combined_df, sample_fraction=0.5, batch_size=8 * 1024).next()
    print 'Loaded validation data'
    confusion_callback = ConfusionMatrix(validation_data=validation_data)

    training_generator = sample_iterator(combined_df)
    model.fit_generator(training_generator,
                        samples_per_epoch=samples_per_epoch,
                        nb_epoch=1024,
                        show_accuracy=True,
                        nb_worker=2,
                        callbacks=[save_callback, log_callback, confusion_callback],
                        validation_data=validation_data
                        )


if __name__ == '__main__':
    main()
