from Layer.Layer import Layer
import tensorflow as tf


class Resize(Layer):
    def __init__(self, name, factor):
        super(Resize, self).__init__(name, True)
        self.factor = factor

    def forward(self, inputs):
        return tf.image.resize_bicubic(inputs, tf.shape(inputs)[1:3] * self.factor)
