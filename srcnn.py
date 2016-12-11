import os
from utils import *
from PIL import Image
import numpy as np

'''
这个文件负责处理所有的SRCNN问题数据的准备,结果的处理等琐碎的工作
'''


def load_data(width=28, height=28, factor=2, size=1000):
    '''
    :param width: LR图片的宽度
    :param height: LR图片的高度
    :param factor: 从LR到HR的放大倍数,长和宽分别放大.
    :param size: 加载的图片数量
    :return: 图片的列表 train_data, test_data, train_label, test_label
    这些列表的类型应当为numpy ndarray
    train_data和test_data的尺寸为[size,width*factor, height*factor, channel]
    train_label和test_label的尺寸为[size, width*factor, height*factor, channel]
    train和test数据不能有重叠
    '''
    hr_width = width * factor
    hr_height = height * factor
    read_data = []
    read_label = []
    dir_list = ["./data/A"]
    for directory in dir_list:
        data, label = walk_and_load_image(directory, (hr_width, hr_height), (width, height))
        read_data.extend(data)
        read_label.extend(label)

    size = int(min(size, len(read_data) / 2))
    train_data = read_data[0:size]
    test_data = read_data[size:size + size]
    train_label = read_label[0:size]
    test_label = read_label[size:size + size]
    # convert type, and read_data become a tensor implicitly
    # resize the images
    # after the process we need to convert it to list

    return [train_data, test_data, train_label, test_label]


def walk_and_load_image(directory, hr_size, lr_size):
    '''
    遍历目录并且得到所有的JPEG图片文件
    :param directory: 要遍历的目录
    :return: 得到的数据集列表
    '''
    data_list = []
    label_list = []
    for dirName, subdirList, fileList in os.walk(directory):
        log("Travelling Directory: %s" % os.path.abspath(dirName))
        for file in fileList:
            with Image.open(os.path.join(dirName, file)) as img:
                lr = (np.asarray(img.resize(lr_size).resize(hr_size)))
                hr = (np.asarray(img.resize(hr_size)))
                if len(lr.shape) != 3 or lr.shape[2] != 3:
                    continue
                else:
                    data_list.append(lr)
                    label_list.append(hr)
        log("Travelled Directory: %s" % os.path.abspath(dirName))
    return data_list, label_list



