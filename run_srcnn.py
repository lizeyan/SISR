from Layer.ReLU import ReLU
from Layer.Convolution import Convolution
from Layer.Resize import Resize
from Layer.Dropout import Dropout
from solve_srcnn import *
from Loss.MSELoss import MSELoss
from srcnn import *
from Network import *

lr_size = (33, 33)
factor = 2
channel = 3
filter_size = (9, 5, 5)
hr_size = lr_size * factor
train_data, train_label = load_data(["./data/Train/"], lr_size[0], lr_size[1], factor=factor, size=1000, channel=channel, filter_size=filter_size)
test_data, test_label = load_data(["./data/Test/Set5"], factor=factor, size=19, channel=channel, filter_size=filter_size)
print(train_data[0].shape)
print(train_label[0].shape)
print(test_data[0].shape)
print(test_label[0].shape)
print("The real size of train data set is: %d" % len(train_data))
print("The real size of test data set is: %d" % len(test_data))
input_placeholder = tf.placeholder(tf.float32)
label_placeholder = tf.placeholder(tf.float32)
keep_prob_placeholder = tf.placeholder(tf.float32)
model = Network()
model.add(Resize('resize', factor))
model.add(Convolution('conv1', filter_size[0], channel, 64, 0.001))
model.add(ReLU('relu1'))
model.add(Convolution('conv2', filter_size[1], 64, 32, 0.001))
model.add(ReLU('relu2'))
model.add(Convolution('conv3', filter_size[2], 32, channel, 0.001))

loss = MSELoss('MSELoss')
optimizer = tf.train.AdamOptimizer(0.0001)
model.compile(input_placeholder, label_placeholder, keep_prob_placeholder, loss, optimizer)
solve_net(model, train_data, train_label, test_data, test_label,
          batch_size=2, max_epoch=1000000, disp_freq=100, test_freq=1000,
          save_path="./model/model_factor2_955_normalized_labelresize/", load_path="./model/model_factor2_955_normalized_labelresize/",
          save_res_freq=10000)


