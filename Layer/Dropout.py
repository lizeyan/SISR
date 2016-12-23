from Layer.Layer import Layer
import tensorflow as tf


class Dropout(Layer):
    def __init__(self, name, keep_prob):
        Layer.__init__(self, name, False)
        self.name = name
        self.keep_prob = keep_prob

    def forward(self, inputs):
        with tf.name_scope(self.name):
            return tf.nn.dropout(inputs, keep_prob=self.keep_prob)
