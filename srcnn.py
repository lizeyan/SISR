import os
from utils import *
from PIL import Image
import numpy as np

'''
这个文件负责处理所有的SRCNN问题数据的准备,结果的处理等琐碎的工作
'''


def load_data(dir_list, width=None, height=None, factor=2, size=1000, channel=1, filter_size=()):
    '''
    :param dir_list: 要遍历的目录列表
    :param width: LR图片的宽度
    :param height: LR图片的高度
    :param factor: 从LR到HR的放大倍数,长和宽分别放大.
    :param size: 加载的图片数量
    :return: 图片的列表 data, label
    这些列表的类型应当为numpy ndarray
    '''
    if width is not None:
        hr_width = width * factor
    else:
        hr_width = None
    if height is not None:
        hr_height = height * factor
    else:
        hr_height = None
    read_data = []
    read_label = []
    size_loss = sum(filter_size) - len(filter_size)
    for d in dir_list:
        if width is not None and height is not None:
            data, label = walk_and_load_image(d, hr_size=(hr_width, hr_height), lr_size=(width, height), length=size, channel=channel, size_loss=size_loss)
        else:
            data, label = walk_and_load_image(d, hr_size=None, lr_size=None, factor=factor, length=size, channel=channel, size_loss=size_loss)
        read_data.extend(data)
        read_label.extend(label)

        if len(read_data) >= size:
            break

    return read_data, read_label


def walk_and_load_image(directory, length, hr_size, lr_size, factor=None, channel=1, size_loss=0):
    '''
    遍历目录并且得到所有的图片文件
    '''
    data_list = []
    label_list = []
    for dirName, subdirList, fileList in os.walk(directory):
        log("Travelling Directory: %s" % os.path.abspath(dirName))
        for file in fileList:
            if not file.endswith(('jpg', 'JPEG', 'png', 'JPG', 'bmp')):
                continue
            with Image.open(os.path.join(dirName, file)) as img:
                if hr_size is None or lr_size is None:
                    lrs = tuple((int(item / factor) for item in img.size))
                    hrs = tuple((item * factor - size_loss for item in lrs))
                    lr = np.asarray(img.resize(lrs))
                    hr = np.asarray(img.resize(hrs))
                    if len(lr.shape) == 3:
                        for left in range(0, lr.shape[2] - channel + 1, channel):
                            data_list.append(lr[:, :, left:left + channel])
                            label_list.append(hr[:, :, left:left + channel])
                    # elif len(lr.shape) == 2:
                    #     data_list.append(lr[:, :, None])
                    #     label_list.append(hr[:, :, None])
                else:
                    for sub_img in crop(img, hr_size[0] - size_loss, hr_size[1] - size_loss, stride=14):
                        hr = (np.asarray(sub_img))
                        lr = (np.asarray(sub_img.resize(lr_size)))
                        if len(lr.shape) == 3:
                            for left in range(0, lr.shape[2] - channel + 1, channel):
                                data_list.append(lr[:, :, left:left + channel])
                                label_list.append(hr[:, :, left:left + channel])
                        # elif len(lr.shape) == 2:
                        #     data_list.append(lr[:, :, None])
                        #     label_list.append(hr[:, :, None])
        log("Travelled Directory: %s" % os.path.abspath(dirName))
        if len(data_list) >= length:
            break
    return data_list, label_list


def crop(image, width, height, stride=None):
    if stride is None:
        stride = width
    img_width, img_height = image.size
    sub_images = []
    for i in range(0, img_width, stride):
        for j in range(0, img_height, stride):
            box = (i, j, i + width, j + height)
            if i + width >= img_width or j + height >= img_height:
                continue
            sub_images.append(image.crop(box))
    return sub_images




