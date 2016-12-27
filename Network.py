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
        self.keep_prob_placeholder = None
        self.sr = None
        self.merged = None

    def add(self, layer):
        self.layer_list.append(layer)

    def compile(self, input_placeholder, label_placeholder, keep_prob_placeholder, loss, optimizer):
        log("Start Compiling")
        x = input_placeholder
        for layer in self.layer_list:
            x = layer.forward(x)
        self.sr = x
        self.input_placeholder = input_placeholder
        self.label_placeholder = label_placeholder
        self.keep_prob_placeholder = keep_prob_placeholder
        self.loss = loss.forward(self.sr, label_placeholder)
        tf.summary.scalar(name="loss", tensor=self.loss)
        tf.summary.scalar(name="max", tensor=tf.reduce_max(self.sr))
        tf.summary.scalar(name="min", tensor=tf.reduce_min(self.sr))
        self.train_step = optimizer.minimize(self.loss)
        self.merged = tf.summary.merge_all()
        log("Compile finished")

