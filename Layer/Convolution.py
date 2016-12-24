from Layer.Layer import Layer
import tensorflow as tf


class Convolution(Layer):
    def __init__(self, name, kernel_size, inputs_dim, num_output, init_std):
        super(Convolution, self).__init__(name, True)
        weight_shape = [kernel_size, kernel_size, inputs_dim, num_output]
        with tf.name_scope(name):
            self.weight = tf.Variable(name="weight",
                                      initial_value=tf.random_normal(shape=weight_shape, stddev=init_std),
                                      trainable=True)
            # self.bias = tf.Variable(name="bias",
            #                         initial_value=tf.zeros(shape=[num_output]),
            #                         trainable=True)

    def forward(self, inputs):
        with tf.name_scope(self.name):
            # return tf.nn.bias_add(tf.nn.conv2d(inputs, self.weight, strides=[1, 1, 1, 1], padding='VALID'),
            #                       self.bias,
            #                       data_format="NHWC")
            return tf.nn.conv2d(inputs, self.weight, strides=[1, 1, 1, 1], padding="VALID")
