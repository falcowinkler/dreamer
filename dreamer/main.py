import data.dataset
import sys
import os
from dreamer.input import *

if __name__ == '__main__':
    tfrecord_filename = os.getcwd() + "/../data/n_maps.tfrecord"
    dataset_file = tfrecord_filename if len(sys.argv) == 1 else sys.argv[1]
    data.dataset.download_if_not_present(dataset_file)
    sess = tf.Session()
    with sess.as_default():
        data_in = inputs(tfrecord_filename)
        print(sess.run([data_in]))