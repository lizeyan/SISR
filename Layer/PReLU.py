from Layer.Layer import Layer
import tensorflow as tf


class PReLU(Layer):
    def __init__(self, name):
        Layer.__init__(self, name, True)
        with tf.name_scope(name):
            self.alpha = tf.Variable(initial_value=tf.zeros(shape=[1]), name="alpha")

    def forward(self, inputs):
        with tf.name_scope(self.name):
            return tf.nn.relu(inputs) + tf.mul(self.alpha, (inputs - abs(inputs))) * 0.5
