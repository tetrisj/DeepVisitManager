from random import random

import numpy as np
from keras.utils.np_utils import to_categorical
from pyspark import StorageLevel

n_desc = 2 * 201 + 2
max_visits_count = 8
states = ('open', 'moved')


def event_features(event, start_time):
    feature = np.array([event.feature.requestVec, event.feature.hrefVec]).reshape(
        (1, n_desc - 2))
    return np.append(feature, event.event.timestamp - start_time)


def multi_helper(combined, max_t):
    start_vector = np.ones((max_t - 1, n_desc)) * -1  # Indicates start of sequence (since we are using stateful RNNs)
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
    X = np.vstack([event_features(event, start_time) for event in events])
    X = X[labels.nonzero()]
    # Add labels to all features (the last one will be removed if needed later after we make windows)
    X = np.hstack([X, good_labels.reshape((good_labels.shape[0], 1))])
    X = np.vstack([start_vector, X])
    return X, good_labels,visits


def learning_set_visits(combined, max_t=32):
    ''' Given a record containing visits and events, creates data of shape n_samples, max_t, n_desc and labels of shape n_samples
        Each sample contains all valid events in the visit up to that point and then the tested event
        The label is the decision for that combination of events
        The idea is to simulate a state machine that decides whether a new event belongs to the visit
    '''
    X, good_labels, _ = multi_helper(combined, max_t)

    # Make train predictions
    Y = to_categorical((good_labels - 1).astype(int), max_visits_count)  # indicator matrix with labels starting at 0

    # Create learning set windows
    X = X.reshape((1, X.shape[0], n_desc))
    X = np.vstack([X[:, i:i + max_t] for i in range(X.shape[1] - max_t + 1)])

    # Remove last label (because this is what we want to predict)
    X[:, -1, -1] = -1

    return X, Y


def learning_set_transfer(combined, max_t=32):
    ''' Given a record containing visits and events, creates data of shape n_samples, max_t, n_desc and labels of shape n_samples
        Each sample contains all valid events in the visit up to that point and then the tested event
        The label is the decision for that combination of events
        The idea is to simulate a state machine that decides whether a new event belongs to the visit
    '''

    X, good_labels, visits = multi_helper(combined, max_t)

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


def concurrent_local_iterator(rdd, concurrency=64, shuffle=True):
    partition_count = rdd.getNumPartitions()
    starts = range(0, partition_count, concurrency)
    if shuffle:
        starts.sort(key=lambda k: random())

    for start_partition in starts:
        end_partition = min(partition_count, start_partition + concurrency)
        partitions = range(start_partition, end_partition)
        rows = rdd.context.runJob(rdd, lambda x: x, partitions)
        for row in rows:
            yield row


def sample_iterator(df, mode, sample_fraction=1.0, batch_size=3 * 1024, loop=True):
    ''' Sample from dataframe then iterate over partitions, yeilding fixed size batches. Loop if `loop==True`
    :param df: dataframe to sample
    :param batch_size: Number of samples in each yielded batch
    '''
    X = None
    Y = None
    samples_rdd = df.map(learning_set(mode))
    samples_rdd.persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    while True:
        if sample_fraction and sample_fraction < 1.0:
            iteration_samples_rdd = samples_rdd.sample(fraction=sample_fraction, withReplacement=True)
        else:
            iteration_samples_rdd = samples_rdd
        samples_iterator = concurrent_local_iterator(iteration_samples_rdd)
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