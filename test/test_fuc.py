import tensorflow.python as tf
import numpy as np
import os
import timeit

tf.enable_eager_execution()


def test_tensordot(W: tf.Tensor, u: tf.Tensor) -> tf.Tensor:
    v = tf.tensordot(u, W, axes=[[2], [0]])
    return v


def test_matmul(W: tf.Tensor, u: tf.Tensor) -> tf.Tensor:
    W_ = W[tf.newaxis, tf.newaxis, ...]
    u_ = u[..., tf.newaxis]
    W_ = tf.tile(W_, [u.shape[0], 1152, 1, 1])
    v = tf.matmul(W_, u_, transpose_a=True)
    return tf.squeeze(v)


def test_einsum(W: tf.Tensor, u: tf.Tensor) -> tf.Tensor:
    return tf.einsum('ij,aki->akj', W, u)


def test_compare():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    batch = 16
    tf.set_random_seed(1)
    W = tf.get_variable('W', shape=(8, 16), dtype=tf.float32, initializer=tf.initializers.random_normal())
    u = tf.get_variable('u', shape=(batch, 1152, 8), dtype=tf.float32, initializer=tf.initializers.random_normal())

    start = timeit.default_timer()
    for i in range(100):
        v1 = test_tensordot(W, u)
    tim = timeit.default_timer()-start
    print("tensordot", tim)

    start = timeit.default_timer()
    for i in range(100):
        v2 = test_matmul(W, u)
    tim = timeit.default_timer()-start
    print("matmul", tim)

    start = timeit.default_timer()
    for i in range(100):
        v3 = test_einsum(W, u)
    tim = timeit.default_timer()-start
    print("einsum", tim)

    print(np.allclose(v1, v2, atol=0.5e-6))
    print(np.allclose(v1, v3, atol=0.5e-6))


def test_u_hat():
    batch = 16
    tf.set_random_seed(1)
    W = tf.get_variable('W', shape=(1152, 10, 8, 16), dtype=tf.float32, initializer=tf.initializers.random_normal())
    u = tf.get_variable('u', shape=(batch, 1152, 8), dtype=tf.float32, initializer=tf.initializers.random_normal())

    u_hat = tf.einsum('jpkq,ijk->ijpq', W, u)
    print(u_hat.shape)
    assert u_hat.shape == [16, 1152, 10, 16]


def test_u_hat_c():
    batch = 16
    tf.set_random_seed(1)
    b = tf.get_variable('b', shape=(1152, 10), dtype=tf.float32, initializer=tf.initializers.random_normal())
    c = tf.nn.softmax(b, name='c')
    u_hat = tf.get_variable('u_hat', shape=(batch, 1152, 10, 16), dtype=tf.float32, initializer=tf.initializers.random_normal())
    s = tf.einsum('jk,ijkq->ikq', c, u_hat)
    print(s.shape)
    assert s.shape == [16, 10, 16]


def test_squash():
    def my_squash(s):
        s_norm = tf.norm_v2(s, axis=-1, keepdims=True)
        s_square_norm = tf.square(s_norm)
        v = (s_square_norm * s)/((1+s_square_norm)*s_norm)
        return v

    def he_squash(s):
        s_square_norm = tf.reduce_sum(tf.square(s), -1, keepdims=True)
        scalar_factor = s_square_norm / (1 + s_square_norm) / (tf.sqrt(s_square_norm) + 1.e-9)
        v = scalar_factor * s  # element-wise
        return v

    def squash(s):
        s_norm = tf.norm_v2(s, axis=-1, keepdims=True)
        s_square_norm = tf.square(s_norm)
        v = (s_norm * s)/(1+s_square_norm)
        return v

    batch = 16
    tf.set_random_seed(1)
    s = tf.get_variable('s', shape=(batch, 10, 16), dtype=tf.float32, initializer=tf.initializers.random_normal())

    v1 = my_squash(s)
    v2 = he_squash(s)
    v3 = squash(s)
    # print(tf.equal(v1, v2))
    # np.allclose(v1.numpy(), v2.numpy())
    np.allclose(v2.numpy(), v3.numpy())


def test_update_b():
    batch = 16
    tf.set_random_seed(1)
    b = tf.get_variable('b', shape=(1152, 10), dtype=tf.float32, initializer=tf.initializers.random_normal())
    u_hat = tf.get_variable('u_hat', shape=(batch, 1152, 10, 16), dtype=tf.float32, initializer=tf.initializers.random_normal())
    v = tf.get_variable('v', shape=(batch, 10, 16), dtype=tf.float32, initializer=tf.initializers.random_normal())

    delat = tf.einsum('iqjk,ijk->qj', u_hat, v)
    b = b+delat
    print(b.shape)
    assert b.shape == [1152, 10]


def test_bias_add():
    batch = 16
    tf.set_random_seed(1)
    bias = tf.get_variable('bias', shape=(10, 16), dtype=tf.float32, initializer=tf.initializers.random_normal())
    v = tf.get_variable('v', shape=(batch, 10, 16), dtype=tf.float32, initializer=tf.initializers.random_normal())
    v = tf.add(v, bias)
    print(v.shape)


def test_consant():
    t = tf.ones((16, 10))
    b = tf.constant(tf.zeros((t.shape[0], t.shape[1])), name='b')  # b [in_caps,out_caps]
