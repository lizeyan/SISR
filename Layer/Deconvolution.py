from Layer.Layer import Layer
import tensorflow as tf


class Deconvolution(Layer):
    def __init__(self, name, kernel_size, inputs_dim, num_output, factor, init_std):
        super(Deconvolution, self).__init__(name, True)
        self.num_output = num_output
        self.factor = factor
        self.kernel_size = kernel_size
        weight_shape = [kernel_size, kernel_size, num_output, inputs_dim]
        with tf.name_scope(name):
            self.weight = tf.Variable(name="weight",
                                      initial_value=tf.random_normal(shape=weight_shape, stddev=init_std),
                                      trainable=True)
            self.bias = tf.Variable(name="bias",
                                    initial_value=tf.zeros(shape=[num_output]),
                                    trainable=True)

    def forward(self, inputs):
        with tf.name_scope(self.name):
            conv = tf.nn.conv2d_transpose(inputs, self.weight,
                                          output_shape=[tf.shape(inputs)[0],
                                                        self.factor * (tf.shape(inputs)[1]),
                                                        self.factor * (tf.shape(inputs)[2]),
                                                        self.num_output],
                                          strides=[1, self.factor, self.factor, 1],
                                          padding="SAME", data_format="NHWC")
            return conv
            # return tf.nn.conv2d(inputs, self.weight, strides=[1, 1, 1, 1], padding="VALID")
