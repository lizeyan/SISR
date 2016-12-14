from Layer.ReLU import ReLU
from Layer.Convolution import Convolution
from Layer.Resize import Resize
from solve_srcnn import *
from Loss.MSELoss import MSELoss
from srcnn import *
from Network import *

lr_size = (32, 32)
factor = 2
channel = 3
filter_size = (9, 3, 5)
hr_size = lr_size * factor
train_data, train_label = load_data(["./data/Train/"], lr_size[0], lr_size[1], factor=factor, size=100000, channel=channel)
test_data, test_label = load_data(["./data/Test/Set5"], factor=factor, size=19, channel=channel)
model = Network()
model.add(Resize('resize', factor))
model.add(Convolution('conv1', filter_size[0], channel, 64, 0.001))
model.add(ReLU('relu1'))
model.add(Convolution('conv2', filter_size[1], 64, 32, 0.001))
model.add(ReLU('relu2'))
model.add(Convolution('conv3', filter_size[2], 32, channel, 0.001))

loss = MSELoss('MSELoss', hr_size[0] - sum(filter_size) + len(filter_size), hr_size[0] - sum(filter_size) + len(filter_size))
optimizer = tf.train.AdamOptimizer(0.001)
input_placeholder = tf.placeholder(tf.float32)
label_placeholder = tf.placeholder(tf.float32)
model.compile(input_placeholder, label_placeholder, loss, optimizer)
solve_net(model, train_data, train_label, test_data, test_label,
          batch_size=32, max_epoch=1000000, disp_freq=100, test_freq=1000,
          save_path="./model/model_factor2_935/", load_path=None,
          save_res_freq=10000)


