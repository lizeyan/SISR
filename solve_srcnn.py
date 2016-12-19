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


def solve_net(model, train_x, train_y, test_x, test_y, batch_size, max_epoch, disp_freq, test_freq,
              save_res_freq, keep_prob=0.5,
              save_path="./model/model/", load_path=None):
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if load_path is None:
        sess.run(tf.initialize_all_variables())
    else:
        ckpt = tf.train.get_checkpoint_state(load_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            log("Load model from %s" % load_path)
        else:
            log("Warning: No checkpoitn found in %s" % load_path)
            sess.run(tf.initialize_all_variables())
    tic = time.time()
    iter_counter = 0
    for k in range(max_epoch):
        for x, y in data_iterator(train_x, train_y, batch_size):
            iter_counter += 1
            model.train_step.run(feed_dict={model.input_placeholder: x,
                                            model.label_placeholder: y})
            if disp_freq is not None and iter_counter % disp_freq == 0:
                sr = model.sr.eval(feed_dict={model.input_placeholder: x,
                                              model.label_placeholder: y})
                psnr = evaluation_PSNR(sr, y)
                log('Iter:%d, train PSNR: %f' % (iter_counter, psnr))
            if test_freq is not None and iter_counter % test_freq == 0:
                log("Testing......")
                test_PSNR = test(model, test_x, test_y, iter_counter % save_res_freq == 0)
                log("Iter:%d, test PSNR: %f" % (iter_counter, test_PSNR))
                saved = saver.save(sess, save_path=save_path, global_step=iter_counter + 1)
                log("Model saved in %s" % saved)

    toc = time.time()
    log("Total train time: %dseconds" % (toc - tic))


def test(model, test_x, test_y, save_output=True):
    test_PSNR = []
    counter = 0
    channel = test_x[0].shape[2]
    for x, y in data_iterator(test_x, test_y, 1, shuffle=False):
        sr = model.sr.eval(feed_dict={model.input_placeholder: x,
                                      model.label_placeholder: y,
                                      model.keep_prob_placeholder: 1.0})
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
                    lr_img = Image.fromarray((x[i]).astype(np.uint8))
                    hr_img = Image.fromarray((y[i]).astype(np.uint8))
                    hr_pdt = Image.fromarray((np.asarray(sr[i])).astype(np.uint8))
                lr_img.save("./test_results/%d_input.jpg" % counter)
                hr_img.save("./test_results/%d_label.jpg" % counter)
                hr_pdt.save("./test_results/%d_predict_%f.jpg" % (counter, psnr))
    return np.mean(test_PSNR)


def evaluation_PSNR(data, label):
    mse = np.mean(np.square(data - label))
    return 20 * np.log10(255) - 10 * np.log10(mse)
