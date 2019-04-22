import tensorflow.python as tf
from tensorflow.python import keras


class CapsDense(keras.layers.Layer):
    def __init__(self,
                 units: int,
                 vec_len: int,
                 activation=None,
                 use_routing=False,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(CapsDense, self).__init__(**kwargs)
        self.units = units
        self.vec_len = vec_len
        self.activation = keras.activations.get(activation)
        self.use_routing = use_routing
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        # input_shape = tf.tensor_shape.TensorShape(input_shape)

        # W [in_caps,out_caps,in_len,out_len]
        self.W = self.add_weight(
            name='W', shape=(input_shape[1].value, self.units, input_shape[2].value, self.vec_len),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=tf.float32,
            trainable=True)

        self.b = self.add_weight(
            name='b', shape=(input_shape[1].value, self.units),
            initializer=keras.initializers.zero(),
            regularizer=None,
            constraint=None,
            dtype=tf.float32,
            trainable=False)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias', shape=[self.units, self.vec_len],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=tf.float32,
                trainable=True)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        # inputs [batch,caps,vec_len]
        if self.use_routing:
            u_hat = tf.einsum('jpkq,ijk->ijpq', self.W, inputs)  # u_hat [batch,in_caps,out_caps,out_len]
            outputs = self.routing(u_hat)
        else:
            outputs = tf.einsum('jpkq,ijk->ipq', W, inputs)  # outputs [batch,out_caps,out_len]
            if self.use_bias:
                outputs = tf.add(outputs, bias)
            if self.activation is not None:
                outputs = self.activation(u_hat)
        return outputs

    def routing(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Dynamic Routing

        Parameters
        ----------
        inputs: tf.Tensor
            shape must be[batch, in_caps, out_cpas, out_len]

        Returns
        -------
        tf.Tensor
            outputs, shape[batch, out_caps, out_len]
        """
        # b = tf.constant(np.zeros(shape=(inputs.shape[1], inputs.shape[2])), name='b')  # b [in_caps,out_caps]
        with tf.variable_scope('routing'):
            with tf.variable_scope('iter_1'):
                c = tf.nn.softmax(self.b)  # c [in_caps,out_caps]
                s = tf.einsum('jk,ijkq->ikq', c, inputs)  # s [batch,out_caps,out_len]
                v = squash(s)  # v [batch,out_caps,out_len]
                self.b = tf.add(self.b, tf.einsum('iqjk,ijk->qj', inputs, v))
            with tf.variable_scope('iter_2'):
                c = tf.nn.softmax(self.b)
                s = tf.einsum('jk,ijkq->ikq', c, inputs)
                v = squash(s)
                self.b = tf.add(self.b, tf.einsum('iqjk,ijk->qj', inputs, v))
            with tf.variable_scope('iter_3'):
                c = tf.nn.softmax(self.b)
                s = tf.einsum('jk,ijkq->ikq', c, inputs)
                if self.use_bias:
                    s = tf.add(s, self.bias)
                v = squash(s)
        return v


def squash(s: tf.Tensor) -> tf.Tensor:
    """squash activation 
        NOTE : euclidean norm is tf.sqrt(tf.square(s))

    $$v_j =\farc{| |s_j||^2}{1+||s_j||^2}\farc{s_j}{||s_j||}$$

    Parameters
    ----------
    s: tf.Tensor
        s shape [batch,caps,len]

    Returns
    -------
    tf.Tensor
        v vector
        v shape equal s
    """
    with tf.variable_scope('squash'):
        s_norm = tf.norm_v2(s, axis=-1, keepdims=True)
        s_square_norm = tf.square(s_norm)
        v = (s_norm * s)/(1+s_square_norm)
        return v


def capsnet(inputs: tf.Tensor):
    layer1 = keras.layers.Conv2D(256, 9, strides=1, padding='valid')
    layer2 = keras.layers.Conv2D(32*8, 9, strides=2)
    reshape1 = keras.layers.Reshape((-1, 8))
    active1 = keras.layers.Activation(squash)
    layer3 = CapsDense(units=10, vec_len=16, activation=squash, use_routing=True, use_bias=True)
    final = keras.layers.Lambda(lambda x: tf.reduce_sum(tf.abs(x), axis=-1)+1.e-9, name="final")
    with tf.variable_scope('CapsNet'):
        x = layer1(inputs)
        x = layer2(x)  # tf.Tensor
        x = reshape1(x)
        x = active1(x)
        caps_out = layer3(x)
        logits = final(caps_out)
    return logits, caps_out


def decoder(caps_out, y):
    with tf.variable_scope('Decoder'):
        mask = tf.einsum('ijk,ij->ik', caps_out, y)
        fc1 = keras.layers.Dense(units=512)(mask)
        fc2 = keras.layers.Dense(units=1024)(fc1)
        decoded = keras.layers.Dense(units=784, activation=tf.nn.sigmoid)(fc2)
    return decoded
