from Loss.Loss import Loss
import tensorflow as tf


class MSELoss(Loss):
    def __init__(self, name):
        super(MSELoss, self).__init__(name)
        # self.target_height = target_height
        # self.target_width = target_width

    def forward(self, x, y):
        # y = tf.slice(y, begin=tf.to_int32(tf.div(tf.shape(y) - tf.shape(x), tf.constant(2))), size=tf.shape(x))
        return tf.reduce_mean(tf.square(x - y))

