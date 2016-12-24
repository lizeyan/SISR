from Layer.Layer import Layer
import tensorflow as tf


class Resize(Layer):
    def __init__(self, name, factor):
        super(Resize, self).__init__(name, True)
        self.factor = factor

    def forward(self, inputs):
        with tf.name_scope(self.name):
            with tf.name_scope(self.name + "/get_shape"):
                shape = tf.shape(inputs)[1:3] * self.factor
            return tf.image.resize_bicubic(inputs, shape)
