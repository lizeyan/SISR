from Loss.Loss import Loss
import tensorflow as tf


class MSELoss(Loss):
    def __init__(self, name, target_height=None, target_width=None):
        super(MSELoss, self).__init__(name)
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, x, y):
        with tf.name_scope(self.name):
            if self.target_height is not None and self.target_width is not None:
                y = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, self.target_height, self.target_width), y)
            return tf.reduce_mean(tf.squared_difference(x, y))

