import tensorflow as tf

def image_cast(img):
    return tf.cast(img * 127.5 + 127.5, tf.uint8)

def kl_loss(avg, log_var):
    with tf.name_scope('KLLoss'):
        return tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + log_var - tf.square(avg) - tf.exp(log_var), axis=-1))

def lrelu(x, alpha=0.02):
    with tf.name_scope('LeakyReLU'):
        return tf.maximum(x, alpha * x)

def binary_accuracy(y_true, y_pred):
    with tf.name_scope('BinaryAccuracy'):
        return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.round(tf.sigmoid(y_pred))), dtype=tf.float32))

def sample_normal(avg, log_var):
    with tf.name_scope('SampleNormal'):
        epsilon = tf.random_normal(tf.shape(avg))
        return tf.add(avg, tf.multiply(tf.exp(0.5 * log_var), epsilon))

def vgg_conv_unit(x, filters, layers, training=True):
    # Convolution
    for i in range(layers):
        x = tf.layers.conv2d(x, filters, (3, 3), (1, 1), 'same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=training)
        x = lrelu(x)

    # Downsample
    x = tf.layers.conv2d(x, filters, (2, 2), (2, 2), 'same',
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=training)
    x = lrelu(x)

    return x

def vgg_deconv_unit(x, filters, layers, training=True):
    # Upsample
    x = tf.layers.conv2d_transpose(x, filters, (2, 2), (2, 2), 'same',
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=training)
    x = lrelu(x)

    # Convolution
    for i in range(layers):
        x = tf.layers.conv2d(x, filters, (3, 3), (1, 1), 'same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=training)
        x = lrelu(x)

    return x

def time_format(t):
    m, s = divmod(t, 60)
    m = int(m)
    s = int(s)
    if m == 0:
        return '%d sec' % s
    else:
        return '%d min %d sec' % (m, s)
