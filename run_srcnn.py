from Layer.ReLU import ReLU
from Layer.Convolution import Convolution
from solve_srcnn import *
from Loss.MSELoss import MSELoss
from srcnn import *
from Network import *

lr_size = (28, 28)
factor = 2
train_data, test_data, train_label, test_label = load_data(width=lr_size[0], height=lr_size[1], factor=factor)
model = Network()
model.add(Convolution('conv1', 9, 3, 64, 0.01))
model.add(ReLU('relu1'))
model.add(Convolution('conv2', 1, 64, 32, 0.01))
model.add(ReLU('relu2'))
model.add(Convolution('conv3', 5, 32, 3, 0.01))

loss = MSELoss('MSELoss')
optimizer = tf.train.AdamOptimizer(0.001)
input_placeholder = tf.placeholder(tf.float32)
label_placeholder = tf.placeholder(tf.float32)
model.compile(input_placeholder, label_placeholder, loss, optimizer)
solve_net(model, train_data, train_label, test_data, test_label,
          batch_size=128, max_epoch=10000, disp_freq=100, test_freq=1000)


