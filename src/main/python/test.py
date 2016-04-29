import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from pyspark import SparkContext, SQLContext, SparkConf
from sklearn.metrics import confusion_matrix


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
    from input import max_visits_count, sample_iterator
    from models import make_model
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

    training_generator = sample_iterator(test_combined_df, mode, loop=False)
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
    plot_confusion_matrix(cm, log_path + '.png')


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
