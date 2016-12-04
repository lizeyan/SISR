from . import Layer
import tensorflow as tf


class ReLU(Layer):
    def __init__(self, name):
        super(ReLU, self).__init__(name, False)

    def forward(self, inputs):
        return tf.nn.relu(inputs)
