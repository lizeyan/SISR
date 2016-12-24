from Layer.Layer import Layer
import tensorflow as tf
import numpy


class Linear(Layer):
    def __init__(self, name, inputs_dim, num_output, init_std):
        super(Linear, self).__init__(name, True)
        with tf.name_scope(name):
            self.weight = tf.Variable(name="weight",
                                      initial_value=tf.random_normal(shape=[inputs_dim, num_output], stddev=init_std),
                                      trainable=True)
            self.bias = tf.Variable(name="bias",
                                    initial_value=tf.zeros(shape=[num_output]),
                                    trainable=True)

    def forward(self, inputs):
        # 将输入flatten到2维
        with tf.name_scope(self.name):
            dim = numpy.prod(inputs.get_shape().as_list()[1:])
            return tf.matmul(tf.reshape(inputs, [-1, dim]), self.weight) + self.bias
