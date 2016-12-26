from Layer.ReLU import ReLU
from Layer.PReLU import PReLU
from Layer.Convolution import Convolution
from Layer.Resize import Resize
from Layer.Dropout import Dropout
from solve_srcnn import *
from Loss.MSELoss import MSELoss
from Loss.Perceptual_Loss import Perceptual_Loss
from srcnn import *
from Network import *

lr_size = (32, 32)
factor = 2
channel = 3
filter_size = (9, 1, 5)
filter_num = (64, 32)
size_loss = sum(filter_size) - len(filter_size)
hr_size = tuple(item * factor for item in lr_size)
print("low resolution size: ", lr_size)
print("high resolution size: ", hr_size)
train_data, train_label = load_data(["./data/Train/Train"], lr_size[0], lr_size[1], factor=factor, size=10000, channel=channel)
print("train data shape", np.shape(train_data))
print("train label shape", np.shape(train_label))
test_data, test_label = load_data(["./data/Test/Set5"], factor=factor, size=5, channel=channel)
print("The real size of train data set is: %d" % len(train_data))
print("The real size of test data set is: %d" % len(test_data))

input_placeholder = tf.placeholder(tf.float32, name="input_dataa")
label_placeholder = tf.placeholder(tf.float32, name="input_label")
keep_prob_placeholder = tf.placeholder(tf.float32, name="keep_prob")

model = Network()
model.add(Resize('resize', factor))
model.add(Convolution('Patch_extraction', filter_size[0], channel, filter_num[0], 0.0001))
model.add(PReLU('prelu1'))
model.add(Convolution('Mapping', filter_size[1], filter_num[0], filter_num[1], 0.0001))
model.add(PReLU('prelu2'))
model.add(Convolution('Reconstruction', filter_size[2], filter_num[1], channel, 0.0001))

loss1 = Perceptual_Loss('Perceptual_Loss')
optimizer = tf.train.AdamOptimizer(0.00001)
model.compile(input_placeholder, label_placeholder, keep_prob_placeholder, loss1, optimizer)
solve_net(model, train_data, train_label, test_data, test_label,
          batch_size=4, max_epoch=1000000, disp_freq=100, test_freq=1000,
          save_path="./model_srcnn_ploss/factor2_915_3/", load_path="./model_srcnn_ploss/factor2_915_3/",
          save_res_freq=1000)


