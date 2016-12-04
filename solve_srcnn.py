import time
import numpy as np
from utils import log
import tensorflow as tf


def data_iterator(x, y, batch_size, shuffle=True):
    length = len(x)
    index = list(range(length))
    if shuffle:
        np.random.shuffle(index)

    for start_idx in range(0, length, batch_size):
        end_idx = min(start_idx + batch_size, length)
        yield x[start_idx:end_idx], y[start_idx:end_idx]


def solve_net(model, train_x, train_y, test_x, test_y, batch_size, max_epoch, disp_freq, test_freq):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tic = time.time()
    iter_counter = 0
    for k in range(max_epoch):
        for x, y in data_iterator(train_x, train_y, batch_size):
            iter_counter += 1
            model.train_step.run(feed_dict={model.input_placeholder: x, model.label_placeholder: y})
            if iter_counter % disp_freq == 0:
                train_PSNR = model.evaluation_PSNR.eval(feed_dict={model.input_placeholder: x, model.label_placeholder: y})
                log('Iter:%d, train PSNR: %f' % (iter_counter, train_PSNR))

    test_PSNR = model.evaluation_PSNR.eval(feed_dict={model.input_placeholder: test_x, model.label_placeholder: test_y})
    log("Final Test PSNR: %f" % test_PSNR)
    toc = time.time()
    log("Total train time: %d" % (toc - tic))

