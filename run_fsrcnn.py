from Layer.ReLU import ReLU
from Layer.PReLU import PReLU
from Layer.Convolution import Convolution
from Layer.Deconvolution import Deconvolution
from Layer.Resize import Resize
from Layer.Dropout import Dropout
from solve_srcnn import *
from Loss.MSELoss import MSELoss
from srcnn import *
from Network import *

lr_size = (11, 11)
factor = 3
channel = 3
filter_size = (5, 1, 3, 1, 9)
filter_num = (56, 16)  # d s
hr_size = tuple(item * factor for item in lr_size)
print("low resolution size: ", lr_size)
print("high resolution size: ", hr_size)
train_data, train_label = load_data(["./data/Train/Set91"], lr_size[0], lr_size[1], factor=factor, size=5000000, channel=channel)
print("train data shape", np.shape(train_data))
print("train label shape", np.shape(train_label))
test_data, test_label = load_data(["./data/Test/"], factor=factor, size=500, channel=channel, resize=False)
print("The real size of train data set is: %d" % len(train_data))
print("The real size of test data set is: %d" % len(test_data))

input_placeholder = tf.placeholder(tf.float32, name="input_dataa")
label_placeholder = tf.placeholder(tf.float32, name="input_label")
keep_prob_placeholder = tf.placeholder(tf.float32, name="keep_prob")

model = Network()
model.add(Convolution(name="Feature_extraction", kernel_size=filter_size[0],
                      inputs_dim=channel, num_output=filter_num[0], init_std=1e-4))
model.add(PReLU('prelu1'))
model.add(Convolution(name="Shrinking", kernel_size=filter_size[1],
                      inputs_dim=filter_num[0], num_output=filter_num[1], init_std=1e-4))
model.add(PReLU('prelu2'))
model.add(Convolution(name="Mapping", kernel_size=filter_size[2],
                      inputs_dim=filter_num[1], num_output=filter_num[1], init_std=1e-4))
model.add(PReLU('prelu3'))
model.add(Convolution(name="Expanding", kernel_size=filter_size[3],
                      inputs_dim=filter_num[1], num_output=filter_num[0], init_std=1e-4))
model.add(PReLU('prelu4'))
model.add(Deconvolution(name="Deconvolution", kernel_size=filter_size[4],
                        inputs_dim=filter_num[0], num_output=channel, init_std=1e-4,
                        factor=factor))

loss = MSELoss('MSELoss')
optimizer = tf.train.AdamOptimizer(0.0001)
# optimizer = tf.train.GradientDescentOptimizer(0.0000001)
model.compile(input_placeholder, label_placeholder, keep_prob_placeholder, loss, optimizer)
solve_net(model, train_data, train_label, test_data, test_label,
          batch_size=4, max_epoch=1000000, disp_freq=100, test_freq=1000,
          save_path="./model_fsrcnn/factor3_51319_3/", load_path="./model_fsrcnn/factor2_51319_3/",
          # save_path="./model_fsrcnn/factor2_test/", load_path=None,
          save_res_freq=10000)


