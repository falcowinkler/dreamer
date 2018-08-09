from dreamer.ops import *

batch_size = 64


def D(x, reuse):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        x = tf.layers.conv2d(x, 32, 4, 2, padding='SAME')
        x = leaky_relu(tf.layers.batch_normalization(x, training=True))
        x = tf.layers.conv2d(x, 64, 4, 2, padding='SAME')
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 2 * 4 * 64)
        x = leaky_relu(tf.layers.batch_normalization(x, training=True))
        x = tf.layers.dense(x, 1, activation=None)
        return tf.nn.sigmoid(x), x


def G(z):
    with tf.variable_scope("generator"):
        z = tf.layers.dense(z, 31 * 23 * 64, activation=tf.nn.relu)
        z = tf.layers.batch_normalization(z, training=True)
        z = tf.reshape(z, [-1, 31, 23, 64])
        z = tf.layers.batch_normalization(z, training=True)
        z = tf.layers.conv2d_transpose(z, 64, 5, 1, padding='SAME', activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        z = tf.layers.batch_normalization(z, training=True)
        z = tf.layers.conv2d_transpose(z, 33, 5, 1, padding='SAME', activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        return z
