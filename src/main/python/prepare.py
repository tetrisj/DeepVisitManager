import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback
from pyspark import SparkContext, SQLContext, SparkConf
from scipy import sparse
import os

import warnings

warnings.filterwarnings("ignore")

# Move to global config
n_desc = 3 * 100 + 1
max_t = 5
# states = ('open', 'moved', 'not-mine', 'closed')
states = ('open', 'moved')


def event_features(event, start_time):
    feature = np.array([event.feature.requestVec, event.feature.hrefVec, event.feature.prevVec]).reshape(
            (1, n_desc - 1))
    return np.append(feature, event.event.timestamp - start_time)


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

    event_data = np.vstack([event_features(event, start_time) for event in combined.events if
                            event.event.timestamp > start_time - 0.01])

    X = np.zeros((n_samples, max_t, n_desc))
    # TODO: use something more efficient than iterations
    for i in range(n_samples):
        X[i][max_t - 1] = event_data[i]
        true_for_i = [idx for idx in true_indices if idx < i][-max_t + 1:]
        if true_for_i:
            X[i][max_t - 1 - len(true_for_i): max_t - 1] = event_data[true_for_i]

    return X, Y


def make_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(max_t, n_desc)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(2, return_sequences=False))
    model.add(Activation('softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model



def concurrent_local_iterator(rdd, concurrency=16):
    partition_count = rdd.getNumPartitions()
    for start_partition in range(0, partition_count, concurrency):
        end_partition = min(partition_count, start_partition + concurrency)
        partitions = range(start_partition, end_partition)
        rows = rdd.context.runJob(rdd, lambda x: x, partitions)
        for row in rows:
            yield row


def sample_iterator(df, batch_size=8192):
    ''' Repatedly sample from dataframe then iterate over partitions, finally yeilding fixed size batches
    :param df: dataframe to sample
    :param batch_size: Number of sample in each yielded batch
    '''
    sample_fraction = 0.1  # Mostly used to keep some randomness while still retaining big enough batches
    X = None
    Y = None
    while True:
        samples_rdd = df.sample(fraction=sample_fraction, withReplacement=True).map(learning_set)
        samples_iterator = concurrent_local_iterator(samples_rdd)
        for x, y in samples_iterator:

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


def main():
    conf = SparkConf().set("spark.sql.shuffle.partitions", "4096") \
        .set("spark.python.worker.memory", "2g") \
        .set("spark.memory.storageFraction", "0.1") \
        .set("spark.ui.showConsoleProgress", "false")

    model_save_path = '/home/jenia/Deep/visit.h5'
    log_path = '/home/jenia/Deep/train.log'
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    combined_df = sqlContext.read.parquet('/home/jenia/Deep/combined_train/')
    samples_per_epoch = 1024 * 128

    save_callback = ModelCheckpoint(filepath=model_save_path, save_best_only=True)
    log_callback = WriteLosses(filepath=log_path)
    #    combinedDF = sqlContext.read.parquet(
    #        '/home/jenia/Deep/combined/part-r-00000-fea4cef7-4875-49b0-a9db-4904a007a852.gz.parquet')

    model = make_model()
    if os.path.isfile(model_save_path):
        print 'Loading model from: %s' % model_save_path
        model.load_weights(model_save_path)

    test_combined_df = sqlContext.read.parquet('/home/jenia/Deep/combined_test/')
    validation_data = sample_iterator(test_combined_df, batch_size=1024 * 16).next()
    print 'Loaded validation data'

    training_generator = sample_iterator(combined_df)
    model.fit_generator(training_generator,
                        samples_per_epoch=samples_per_epoch,
                        nb_epoch=200,
                        show_accuracy=True,
                        nb_worker=2,
                        callbacks=[save_callback, log_callback],
                        validation_data=validation_data
                        )


if __name__ == '__main__':
    main()
