import time
import numpy as np
import os
from utils import log
import tensorflow as tf
from PIL import Image


def data_iterator(x, y, batch_size, shuffle=True):
    length = len(x)
    index = list(range(length))
    if shuffle:
        np.random.shuffle(index)

    for start_idx in range(0, length, batch_size):
        end_idx = min(start_idx + batch_size, length)
        yield x[start_idx:end_idx], y[start_idx:end_idx]


def solve_net(model, train_x, train_y, test_x, test_y, batch_size, max_epoch, disp_freq, test_freq, save_res_freq,
              save_path="./model/model/", load_path=None):
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if load_path is None:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, load_path)
        log("Load model from %s" % load_path)
    tic = time.time()
    iter_counter = 0
    for k in range(max_epoch):
        for x, y in data_iterator(train_x, train_y, batch_size):
            iter_counter += 1
            model.train_step.run(feed_dict={model.input_placeholder: x, model.label_placeholder: y})
            if disp_freq is not None and iter_counter % disp_freq == 0:
                sr = model.sr.eval(feed_dict={model.input_placeholder: x, model.label_placeholder: y})
                psnr = evaluation_PSNR(sr, y)
                log('Iter:%d, train PSNR: %f' % (iter_counter, psnr))
            if test_freq is not None and iter_counter % test_freq == 0:
                log("Testing......")
                test_PSNR = test(model, test_x, test_y, iter_counter % save_res_freq == 0)
                log("Iter:%d, test PSNR: %f" % (iter_counter, test_PSNR))
                saved = saver.save(sess, save_path=save_path)
                log("Model saved in %s" % saved)

    toc = time.time()
    log("Total train time: %dseconds" % (toc - tic))


def test(model, test_x, test_y, save_output=True):
    test_PSNR = []
    counter = 0
    channel = test_x[0].shape[2]
    for x, y in data_iterator(test_x, test_y, 1, shuffle=False):
        sr = model.sr.eval(feed_dict={model.input_placeholder: x, model.label_placeholder: y})
        psnr = evaluation_PSNR(sr, y)
        test_PSNR.append(psnr)
        if save_output:
            for i in range(len(x)):
                counter += 1
                if channel == 1:
                    lr_img = Image.fromarray(np.squeeze(x[i], axis=(2, )))
                    hr_img = Image.fromarray(np.squeeze(y[i], axis=(2, )))
                    hr_pdt = Image.fromarray(np.squeeze(np.asarray(sr[i]).astype(np.uint8), axis=(2,)))
                else:
                    lr_img = Image.fromarray(x[i])
                    hr_img = Image.fromarray(y[i])
                    hr_pdt = Image.fromarray(np.asarray(sr[i]).astype(np.uint8))
                lr_img.save("./test_results/%d_input.jpg" % counter)
                hr_img.save("./test_results/%d_label.jpg" % counter)
                hr_pdt.save("./test_results/%d_predict_%f.jpg" % (counter, psnr))
    return np.mean(test_PSNR)


def evaluation_PSNR(data, label):
    data = np.asarray(data)
    label = np.asarray(label)
    width_data, height_data = data.shape[1:3]
    width_label, height_label = label.shape[1:3]
    width = min(width_data, width_label)
    height = min(height_data, height_label)
    border_data_width = int((width_data - width) / 2)
    border_data_height = int((height_data - height) / 2)
    border_label_width = int((width_label - width) / 2)
    border_label_height = int((height_label - height) / 2)
    data_center = data[:, border_data_width:border_data_width + width, border_data_height:border_data_height + height, :]
    label_center = label[:, border_label_width:border_label_width + width, border_label_height:border_label_height + height, :]
    mse = np.mean((data_center - label_center) ** 2)
    return 20 * np.log10(255) - 10 * np.log10(mse)
