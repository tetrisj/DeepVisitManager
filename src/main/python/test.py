import os
import warnings

import numpy as np
import argparse
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from pyspark import SparkContext, SQLContext, SparkConf
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
# Move to global config
n_desc = 3 * 201 + 2
max_visits_count = 16
max_t = 32
states = ('open', 'moved')
start_vector = np.ones((max_t - 1, n_desc)) * -1  # Indicates start of sequence (since we are using stateful RNNs)


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

    n_samples = len(events)
    event_timestamps = np.array([event.event.timestamp for event in events])
    start_time = event_timestamps.min()
    labels = np.zeros(n_samples)
    # Calculate labels
    for i, visit in enumerate(visits[:max_visits_count]):
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
    Y = to_categorical((good_labels - 1).astype(int), max_visits_count)  # indicator matrix with labels starting at 0

    # Create learning set windows
    X = X.reshape((1, X.shape[0], n_desc))
    X = np.vstack([X[:, i:i + max_t] for i in range(X.shape[1] - max_t + 1)])

    # Remove last label (because this is what we want to predict)
    X[:, -1, -1] = -1

    return X, Y


def learning_set_transfer(combined):
    ''' Creates data of shape n_samples, max_t, n_desc and labels of shape n_samples
        Each sample contains all valid events in the visit up to that point and then the tested event
        The label is the decision for that combination of events
        The idea is to simulate a state machine that decides whether a new event belongs to the visit
    '''

    events = sorted(combined.events, key=lambda e: e.event.timestamp)
    visits = sorted(combined.visits, key=lambda v: v.timestamps[0])
    n_samples = len(events)
    event_timestamps = np.array([event.event.timestamp for event in events])
    start_time = event_timestamps.min()
    labels = np.zeros(n_samples)
    # Calculate labels for events in the first max_visits_count visits. The rest will be discarded (label 0)
    for i, visit in enumerate(visits[:max_visits_count]):
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
    Y = to_categorical((sending_map[(good_labels - 1).astype(int)]).astype(int),
                       max_visits_count)  # indicator matrix with labels starting at 0

    # Create learning set windows
    X = X.reshape((1, X.shape[0], n_desc))
    X = np.vstack([X[:, i:i + max_t] for i in range(X.shape[1] - max_t + 1)])

    return X, Y

def learning_set(mode):
    if mode == 'visit':
        raise NotImplementedError
    elif mode == 'multi-visit':
        return learning_set_visits
    elif mode == 'multi-transfer':
        return learning_set_transfer

def make_multi_model():
    model = Sequential()
    model.add(LSTM(256, return_sequences=True,
                   input_shape=(max_t, n_desc)))
    model.add(LSTM(128, return_sequences=False,
                   input_shape=(max_t, n_desc)))
    model.add(Dense(max_visits_count))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
    # sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer=sgd)
    return model

def make_visit_model():
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


def concurrent_local_iterator(rdd, concurrency=64):
    partition_count = rdd.getNumPartitions()
    for start_partition in range(0, partition_count, concurrency):
        end_partition = min(partition_count, start_partition + concurrency)
        partitions = range(start_partition, end_partition)
        rows = rdd.context.runJob(rdd, lambda x: x, partitions)
        for row in rows:
            yield row


def sample_iterator(df, mode, sample_fraction=1.0, batch_size=3 * 1024, loop = False):
    ''' Repatedly sample from dataframe then iterate over partitions, finally yeilding fixed size batches
    :param df: dataframe to sample
    :param batch_size: Number of samples in each yielded batch
    '''
    X = None
    Y = None
    learning_set_function = learning_set(mode)
    while True:
        if sample_fraction and sample_fraction<1.0:
            samples_rdd = df.sample(fraction=sample_fraction, withReplacement=True)
        else:
            samples_rdd = df
        samples_rdd = samples_rdd.map(learning_set(mode))
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
        if not loop:
            if X is not None:
                yield X, Y
            return

def plot_confusion_matrix(cm, path):
    title = 'Confusion matrix'
    cmap = plt.cm.Blues
    plt.figure()
    plt.imshow(cm / float(cm.sum()), interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)


def main(mode, test_data_path, model_path, log_path):
    conf = SparkConf().set("spark.sql.shuffle.partitions", "4096") \
        .set("spark.python.worker.memory", "2g") \
        .set("spark.memory.storageFraction", "0.1") \
        .set("spark.ui.showConsoleProgress", "false")

    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    model = make_model(mode)
    if os.path.isfile(model_path):
        print 'Loading model from: %s' % model_path
        model.load_weights(model_path)

    test_combined_df = sqlContext.read.parquet(test_data_path)

    training_generator = sample_iterator(test_combined_df, mode)
    cm = np.zeros((max_visits_count, max_visits_count), dtype=int)
    test_size = 0
    for X, Y in training_generator:
        P = model.predict_on_batch(X)
        cm += confusion_matrix(np.argmax(P, 1), np.argmax(Y, 1), range(max_visits_count))
        test_size += X.shape[0]

    print 'Confusion Matrix:'
    print cm
    print '-' * 20
    print 'Accuracy: %s' % (cm.diagonal().sum() / float(cm.sum()))
    print 'Samples: %s' % test_size
    plot_confusion_matrix(cm, log_path+'.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', choices=['visit', 'multi-visit', 'multi-transfer'],
                        default='multi-transfer', help='training mode')

    parser.add_argument('--test-path', dest='test_path', default='/home/jenia/Deep/combined_test/',
                        help='Test data path')
    parser.add_argument('--model-path', dest='model_path', default='/home/jenia/Deep/transfer.h5',
                        help='Path of model to load')
    parser.add_argument('--log-path', dest='log_path', default='/home/jenia/Deep/test.log',
                        help='Path to loss/accuracy log file')
    args = parser.parse_args()
    main(args.mode, args.test_path, args.model_path, args.log_path)
