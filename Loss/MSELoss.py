from Loss.Loss import Loss
import tensorflow as tf


class MSELoss(Loss):
    def __init__(self, name):
        super(MSELoss, self).__init__(name)

    def forward(self, x, y):
        with tf.name_scope(self.name):
            return tf.reduce_mean(tf.squared_difference(x, y))

