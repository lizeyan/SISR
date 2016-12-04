from Loss.Loss import Loss
import tensorflow as tf

class MSELoss(Loss):
    def __init__(self, name):
        super(MSELoss, self).__init__(name)

    def forward(self, x, y):
        x_shaped = tf.image.resize_images(x, tf.shape(y)[1:3])
        return tf.reduce_mean(tf.squared_difference(x_shaped, y))

