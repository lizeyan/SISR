from Layer.Layer import Layer
import tensorflow as tf


class ReLU(Layer):
    def __init__(self, name):
        Layer.__init__(self, name, False)

    def forward(self, inputs):
        with tf.name_scope(self.name):
            return tf.nn.relu(inputs)
