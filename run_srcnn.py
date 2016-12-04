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
model.add(Convolution('conv1', 7, 3, 8, 0.01))
model.add(ReLU('relu1'))
model.add(Convolution('conv2', 5, 8, 16, 0.01))
model.add(ReLU('relu2'))

loss = MSELoss('MSELoss')
optimizer = tf.train.AdamOptimizer(1e-4)
input_placeholder = tf.placeholder(tf.float32)
label_placeholder = tf.placeholder(tf.float32)
model.compile(input_placeholder, label_placeholder, loss, optimizer)
solve_net(model, train_data, train_label, test_data, test_label, batch_size=32, max_epoch=100, disp_freq=100, test_freq=1000)


