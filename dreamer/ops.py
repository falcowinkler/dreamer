import tensorflow as tf


def gan_loss(logits_real, logits_fake):
    ones = tf.ones_like(logits_fake)
    zeros = tf.zeros_like(logits_fake)
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=ones))
    D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=ones)) + \
             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=zeros))
    return D_loss, G_loss


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


def sample_noise(batch_size, dim):
    return tf.random_uniform(shape=[batch_size, dim], minval=-1, maxval=1)


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session
