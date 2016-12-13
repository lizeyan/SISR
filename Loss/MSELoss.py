from Loss.Loss import Loss
import tensorflow as tf


class MSELoss(Loss):
    def __init__(self, name):
        super(MSELoss, self).__init__(name)

    @staticmethod
    def forward(x, y):
        return tf.reduce_mean(tf.squared_difference(x, y))

