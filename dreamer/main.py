import data.dataset
import sys
import os
from dreamer.input import *
from dreamer.gan import *
from dreamer.output import *

if __name__ == '__main__':
    tfrecord_filename = os.getcwd() + "/../data/n_maps.tfrecord"
    dataset_file = tfrecord_filename if len(sys.argv) == 1 else sys.argv[1]
    data.dataset.download_if_not_present(dataset_file)
    sess = tf.Session()
    with sess.as_default():

        tf.train.start_queue_runners(sess)
        data_in = inputs(tfrecord_filename)
        G_sample = G(sample_noise(batch_size=batch_size, dim=100))

        D_real, D_logit_real = D(data_in, reuse=False)
        D_fake, D_logit_fake = D(G_sample, reuse=True)

        D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        G_loss = -tf.reduce_mean(tf.log(D_fake))

        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=D_vars)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=G_vars)

        sess.run(tf.global_variables_initializer())
        show_every = 100
        for i in range(10000):
            _, D_loss_curr = sess.run([D_solver, D_loss])
            _, G_loss_curr = sess.run([G_solver, G_loss])
            print("D_loss: " + str(D_loss_curr))
            print("G_loss: " + str(G_loss_curr))
            print("Iteration " + str(i))
            if i % show_every == 0:

                sample = sess.run([tf.argmax(tf.nn.softmax(G_sample), axis=3)])
                to_protobuf(sample, i)