from Layer.ReLU import ReLU
from Layer.Linear import Linear
from Layer.Convolution import Convolution
import tensorflow as tf
from solve_srcnn import *
from Loss.MSELoss import MSELoss
from srcnn import *
from Network import *

train_data, test_data, train_label, test_label = load_data()
model = Network()
model.add(Convolution('conv1', 9, 3, 64, 0.01))
model.add(ReLU('relu1'))
model.add(Convolution('conv2', 1, 64, 32, 0.01))
model.add(ReLU('relu2'))
model.add(Convolution('conv3', 5, 32, 3, 0.01))

loss = MSELoss('MSELoss')
optimizer = tf.train.AdamOptimizer(0.01)
input_placeholder = tf.placeholder(tf.float32)
label_placeholder = tf.placeholder(tf.float32)
model.compile(input_placeholder, label_placeholder, loss, optimizer)
solve_net(model, train_data, train_label, test_data, test_label,
          batch_size=32, max_epoch=10000, disp_freq=100, test_freq=1000)


