import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'raw_tile_data': tf.FixedLenFeature([], tf.string),
        })
    tile_data = tf.decode_raw(features['raw_tile_data'], tf.uint8)
    return tile_data


def inputs(filename, batch_size=64, num_epochs=10):
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(decode)
        dataset = dataset.shuffle(1000 + 3 * batch_size)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
