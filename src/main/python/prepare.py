import argparse
import os
import warnings

from keras.callbacks import ModelCheckpoint, Callback
from pyspark import SparkContext, SQLContext, SparkConf
from sklearn.metrics import confusion_matrix
from input import *
from models import *

warnings.filterwarnings("ignore")


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


def main(mode, train_data_path, test_data_path, model_save_path, log_path):
    conf = SparkConf().set("spark.sql.shuffle.partitions", "4096") \
        .set("spark.python.worker.memory", "2g") \
        .set("spark.memory.storageFraction", "0.1") \
        .set("spark.ui.showConsoleProgress", "false")

    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    combined_df = sqlContext.read.parquet(train_data_path)
    samples_per_epoch = 1024 * 128

    save_callback = ModelCheckpoint(filepath=model_save_path, save_best_only=False)
    log_callback = WriteLosses(filepath=log_path)

    model = make_model(mode)
    if os.path.isfile(model_save_path):
        print 'Loading model from: %s' % model_save_path
        model.load_weights(model_save_path)

    test_combined_df = sqlContext.read.parquet(test_data_path)
    validation_data = sample_iterator(test_combined_df, mode, sample_fraction=1.0, batch_size=8 * 1024).next()
    print 'Loaded validation data'
    confusion_callback = ConfusionMatrix(validation_data=validation_data)

    training_generator = sample_iterator(combined_df, mode, sample_fraction=0.3)
    model.fit_generator(training_generator,
                        samples_per_epoch=samples_per_epoch,
                        nb_epoch=256,
                        show_accuracy=True,
                        callbacks=[save_callback, log_callback, confusion_callback],
                        validation_data=validation_data,
                        max_q_size=samples_per_epoch
                        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', choices=['visit', 'multi-visit', 'multi-transfer'],
                        default='multi-transfer', help='training mode')
    parser.add_argument('--train-path', dest='train_path', default='/home/jenia/Deep/combined_train/',
                        help='Training data path')
    parser.add_argument('--test-path', dest='test_path', default='/home/jenia/Deep/combined_test/',
                        help='Test data path')
    parser.add_argument('--model-path', dest='model_path', default='/home/jenia/Deep/transfer.h5',
                        help='Path of model file to save/load')
    parser.add_argument('--log-path', dest='log_path', default='/home/jenia/Deep/train.log',
                        help='Path to loss/accuracy log file')
    args = parser.parse_args()
    main(args.mode, args.train_path, args.test_path, args.model_path, args.log_path)
