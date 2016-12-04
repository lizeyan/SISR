import tensorflow as tf
from utils import *


class Network(object):
    def __init__(self):
        self.layer_list = []
        self.train_step = None
        self.loss = None
        self.evaluation_PSNR = None
        self.input_placeholder = None
        self.label_placeholder = None

    def add(self, layer):
        self.layer_list.append(layer)

    def compile(self, input_placeholder, label_placeholder, loss, optimizer):
        log("Start Compiling")
        x = input_placeholder
        for layer in self.layer_list:
            x = layer.forward(x)
        self.input_placeholder = input_placeholder
        self.label_placeholder = label_placeholder
        self.loss = loss.forward(x, label_placeholder)
        self.train_step = optimizer.minimize(self.loss)
        self.evaluation_PSNR = 10 * tf.log(1 / tf.reduce_mean(tf.squared_difference(tf.image.resize_images(x, tf.shape(label_placeholder)[1:3]), label_placeholder)))
        log("Compile finished")

