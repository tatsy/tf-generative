import tensorflow as tf

def kl_loss(avg, log_var):
    with tf.name_scope('KLLoss'):
        return tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + log_var - tf.square(avg) - tf.exp(log_var), axis=-1))

def lrelu(x, alpha=0.2):
    with tf.name_scope('LeakyReLU'):
        return tf.maximum(x, alpha * x)

def binary_accuracy(y_true, y_pred):
    with tf.name_scope('BinaryAccuracy'):
        return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.round(tf.sigmoid(y_pred))), dtype=tf.float32))

def sample_normal(avg, log_var):
    with tf.name_scope('SampleNormal'):
        epsilon = tf.random_normal(tf.shape(avg))
        return tf.add(avg, tf.multiply(tf.exp(0.5 * log_var), epsilon))

def time_format(t):
    m, s = divmod(t, 60)
    m = int(m)
    s = int(s)
    if m == 0:
        return '%d sec' % s
    else:
        return '%d min %d sec' % (m, s)