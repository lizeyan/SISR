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

    sx = x[index]
    sy = y[index]
    for start_idx in range(0, length, batch_size):
        end_idx = min(start_idx + batch_size, length)
        yield sx[start_idx:end_idx], sy[start_idx:end_idx]


def solve_net(model, train_x, train_y, test_x, test_y, batch_size, max_epoch, disp_freq, test_freq,
              save_res_freq, keep_prob=0.5, summary_dir="./summary",
              save_path="./model/model/", load_path=None, test_only=False):
    saver = tf.train.Saver()
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    param_counter = tf.zeros([1], dtype=tf.int32)
    for variable in tf.trainable_variables():
        param_counter += tf.size(variable)
    log("The number of trainable parameters: %d" % param_counter.eval())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if load_path is None:
        sess.run(tf.global_variables_initializer())
    else:
        ckpt = tf.train.get_checkpoint_state(load_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            log("Load model from %s" % load_path)
        else:
            log("Warning: No checkpoitn found in %s" % load_path)
            sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
    tic = time.time()
    if test_only:
        test(sess, model, test_x, test_y, True)
    else:
        loss_list = []
        iter_counter = 0
        for k in range(max_epoch):
            for x, y in data_iterator(train_x, train_y, batch_size):
                iter_counter += 1
                summary, _, loss, sr = sess.run([model.merged, model.train_step, model.loss, model.sr],
                                                feed_dict={
                                                    model.input_placeholder: x,
                                                    model.label_placeholder: y})
                loss_list.append(loss)
                if disp_freq is not None and iter_counter % disp_freq == 0:
                    mean_loss = np.mean(loss_list)
                    psnr = evaluation_psnr(mean_loss)
                    log('Iter:%d, train PSNR: %f, mean loss: %f' % (iter_counter, psnr, mean_loss))
                    summary_writer.add_summary(summary, iter_counter)
                if test_freq is not None and iter_counter % test_freq == 0:
                    log("Testing......")
                    test_PSNR, test_time = test(sess, model, test_x, test_y,
                                                save_res_freq is not None and iter_counter % save_res_freq == 0)
                    log("Iter:%d, test PSNR: %f, test FPS: %f fps" % (iter_counter, test_PSNR, 1.0 / test_time))
                    saved = saver.save(sess, save_path=save_path, global_step=iter_counter + 1)
                    log("Model saved in %s" % saved)
                    loss_list.clear()

    toc = time.time()
    log("Total elapsed time: %dseconds" % (toc - tic))


def test(sess, model, test_x, test_y, save_output=True):
    test_psnr = []
    time_list = []
    counter = 0
    channel = test_x[0].shape[2]
    for x, y in data_iterator(test_x, test_y, 2, shuffle=False):
        tic = time.time()
        sr = sess.run(model.sr, feed_dict={model.input_placeholder: x, model.label_placeholder: y})
        toc = time.time()
        time_list.append(toc - tic)
        loss = evaluation_mse(sr, y)
        psnr = evaluation_psnr(loss)
        test_psnr.append(psnr)
        if save_output:
            for i in range(len(x)):
                counter += 1
                if channel == 1:
                    lr_img = Image.fromarray(np.squeeze(x[i], axis=(2,)))
                    hr_img = Image.fromarray(np.squeeze(y[i], axis=(2,)))
                    hr_pdt = Image.fromarray(np.squeeze(np.asarray(sr[i]).astype(np.uint8), axis=(2,)))
                else:
                    lr_img = Image.fromarray((x[i]).astype(np.uint8))
                    hr_img = Image.fromarray((y[i]).astype(np.uint8))
                    hr_pdt = Image.fromarray((np.asarray(sr[i])).astype(np.uint8))
                lr_img.save("./test_results/%d_input.jpg" % counter)
                hr_img.save("./test_results/%d_label.jpg" % counter)
                hr_pdt.save("./test_results/%d_predict_%f.jpg" % (counter, psnr))
    return np.mean(test_psnr), np.mean(time_list)


def evaluation_mse(data, label):
    data = np.asarray(data)
    label = np.asarray(label)
    assert data.shape == label.shape, "data and label have different shape"
    return np.mean((data - label) ** 2)


def evaluation_psnr(loss):
    return 20 * np.log10(255) - 10 * np.log10(loss)
