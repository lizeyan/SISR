from Layer.ReLU import ReLU
from Layer.Convolution import Convolution
from Layer.Resize import Resize
from solve_srcnn import *
from Loss.MSELoss import MSELoss
from srcnn import *
from Network import *

lr_size = (32, 32)
factor = 2
hr_size = lr_size * factor
train_data, train_label = load_data(["./data/Train/"], lr_size[0], lr_size[1], factor=factor, size=100000)
test_data, test_label = load_data(["./data/Test"], factor=factor, size=19)
model = Network()
model.add(Resize('resize', factor))
model.add(Convolution('conv1', 9, 3, 32, 0.001))
model.add(ReLU('relu1'))
model.add(Convolution('conv2', 3, 32, 16, 0.001))
model.add(ReLU('relu2'))
model.add(Convolution('conv3', 5, 16, 3, 0.001))

loss = MSELoss('MSELoss')
optimizer = tf.train.AdamOptimizer(0.001)
input_placeholder = tf.placeholder(tf.float32)
label_placeholder = tf.placeholder(tf.float32)
model.compile(input_placeholder, label_placeholder, loss, optimizer)
solve_net(model, train_data, train_label, test_data, test_label,
          batch_size=32, max_epoch=10000, disp_freq=100, test_freq=1000,
          save_path="./model/model1.ckpt", load_path="./model/model1.ckpt")


