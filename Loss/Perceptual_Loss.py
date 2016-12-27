from Loss.Loss import Loss
import tensorflow as tf
from VGG16.vgg16 import vgg16


class Perceptual_Loss(Loss):
    def __init__(self, name):
        super(Perceptual_Loss, self).__init__(name)
        self.sess1 = tf.Session()
        self.sess2 = tf.Session()
        self.vgg1 = None
        self.vgg2 = None

    def forward(self, x, y):
        self.vgg1 = vgg16(x, 'vgg16_weights.npz', self.sess1)
        self.vgg2 = vgg16(y, 'vgg16_weights.npz', self.sess2)
        x1 = self.vgg1.conv1_2
        y1 = self.vgg2.conv1_2
        return tf.reduce_mean(tf.square(x1 - y1)) + tf.reduce_mean(tf.square(x - y))
