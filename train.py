import tensorflow.python as tf
from tensorflow.python import keras
from model import capsnet, decoder
from tqdm import tqdm


def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = x[..., tf.newaxis]
    y = tf.cast(y, tf.float32)
    return x, y


def mnist_dataset():
    (x, y), (x_val, y_val) = keras.datasets.mnist.load_data()
    y = tf.one_hot(y, depth=10)
    y_val = tf.one_hot(y_val, depth=10)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.apply(tf.data.experimental.map_and_batch(prepare_mnist_features_and_labels, 100, drop_remainder=True))
    ds = ds.apply(tf.data.experimental.shuffle_and_repeat(10000, count=None))

    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.apply(tf.data.experimental.map_and_batch(prepare_mnist_features_and_labels, 100, drop_remainder=True))
    ds_val = ds_val.apply(tf.data.experimental.shuffle_and_repeat(10000, count=None,))

    return ds, ds_val


def calc_loss(logits: tf.Tensor, caps_out: tf.Tensor, x: tf.Tensor, y: tf.Tensor, decoded: tf.Tensor):
    with tf.variable_scope('calc_loss'):
        # margin loss 中调节上margin和下margind的权重
        lambda_val = 0.5
        # 上margin与下margin的参数值
        m_plus = 0.95
        m_minus = 0.05
        max_l = tf.square(tf.maximum(0., m_plus-logits))
        max_r = tf.square(tf.maximum(0., logits-m_minus))

        margin_loss = tf.reduce_mean(tf.reduce_sum(y * max_l + lambda_val * (1. - y) * max_r, axis=-1))

        orgin = tf.reshape(x, (x.shape[0], -1))
        reconstruct_loss = 0.0005*tf.reduce_mean(tf.square(orgin-decoded))
        total_loss = margin_loss+reconstruct_loss
    return total_loss


if __name__ == "__main__":
    g = tf.get_default_graph()
    ds, ds_val = mnist_dataset()
    iterator = ds.make_one_shot_iterator()
    next_x, next_y = iterator.get_next()
    batch_x = tf.placeholder_with_default(next_x, shape=[100, 28, 28, 1])
    batch_y = tf.placeholder_with_default(next_y, shape=[100, 10])
    logits, caps_out = capsnet(batch_x)
    decoded = decoder(caps_out, batch_y)
    """ define loss """
    loss = calc_loss(logits, caps_out, batch_x, batch_y, decoded)
    """ define summary """
    acc_op, acc = tf.metrics.accuracy(tf.argmax(batch_y, -1), tf.argmax(logits, -1))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    tf.summary.image('reconstruction_img', tf.reshape(decoded, (100, 28, 28, 1)))
    summ = tf.summary.merge_all()
    """ define train op """
    steps = tf.train.get_or_create_global_step(g)
    train_op = tf.train.AdamOptimizer().minimize(loss, global_step=steps)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('log', g)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        for i in range(10):
            with tqdm(total=60000//100, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}{postfix}', unit=' batch', dynamic_ncols=True) as t:
                for j in range(60000//100):
                    _, summ_, steps_, loss_, acc_ = sess.run([train_op, summ, steps, loss, acc])
                    t.set_postfix(loss='{:<5.3f}'.format(loss_), acc='{:<4.2f}%'.format(acc_*100))
                    writer.add_summary(summ_, steps_)
                    t.update()
