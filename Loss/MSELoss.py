from Loss.Loss import Loss
import tensorflow as tf


class MSELoss(Loss):
    def __init__(self, name, target_height, target_width):
        super(MSELoss, self).__init__(name)
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, x, y):
        with tf.name_scope(self.name):
            # begin = tf.to_int32(tf.div(tf.shape(y) - tf.shape(x), tf.constant(2)))
            # shape = tf.shape(y, name="shape_y")
            # tf.summary.scalar("begin_0", begin[0])
            # tf.summary.scalar("begin_1", begin[1])
            # tf.summary.scalar("begin_2", begin[2])
            # tf.summary.scalar("begin_3", begin[3])
            # tf.summary.scalar("size_0", shape[0])
            # tf.summary.scalar("size_1", shape[1])
            # tf.summary.scalar("size_2", shape[2])
            # tf.summary.scalar("size_3", shape[3])
            # y = tf.slice(y, begin=begin, size=shape)
            y = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, self.target_height, self.target_width), y)
            # y = tf.image.resize_bicubic(y, size=[self.target_height, self.target_width])
            return tf.reduce_mean(tf.squared_difference(x, y))

