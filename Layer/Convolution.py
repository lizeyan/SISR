from Layer.Layer import Layer
import tensorflow as tf


class Convolution(Layer):
    def __init__(self, name, kernel_size, inputs_dim, num_output, init_std):
        super(Convolution, self).__init__(name, True)
        weight_shape = [kernel_size, kernel_size, inputs_dim, num_output]
        with tf.name_scope(name):
            self.weight = tf.Variable(name="weight",
                                      initial_value=tf.random_normal(shape=weight_shape, stddev=init_std))
            self.bias = tf.Variable(name="bias",
                                    initial_value=tf.zeros(shape=[num_output]))

    def forward(self, inputs):
        return tf.nn.conv2d(inputs, self.weight, strides=[1, 1, 1, 1], padding='SAME') + self.bias
