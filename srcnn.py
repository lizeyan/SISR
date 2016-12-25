import os
from utils import *
from PIL import Image
import numpy as np

'''
这个文件负责处理所有的SRCNN问题数据的准备,结果的处理等琐碎的工作
'''


def load_data(dir_list, width=None, height=None, factor=2, size=1000, channel=1, resize=False):
    '''
    :param dir_list: 要遍历的目录列表
    :param width: LR图片的宽度
    :param height: LR图片的高度
    :param factor: 从LR到HR的放大倍数,长和宽分别放大.
    :param size: 加载的图片数量
    :param channel
    :param resize
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
    for d in dir_list:
        if width is not None and height is not None:
            data, label = walk_and_load_image(d,
                                              hr_size=(hr_width, hr_height),
                                              lr_size=(width, height),
                                              factor=factor,
                                              length=size,
                                              channel=channel,
                                              resize=resize)
        else:
            data, label = walk_and_load_image(d, hr_size=None,
                                              lr_size=None, factor=factor,
                                              length=size, channel=channel,
                                              resize=False)
        read_data.extend(data)
        read_label.extend(label)

        if len(read_data) >= size:
            break

    return np.asarray(read_data), np.asarray(read_label)


def walk_and_load_image(directory, length, hr_size, lr_size, factor=None, channel=1, resize=False):
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
                img_width, img_height = img.size
                if hr_size is None or lr_size is None:
                    lrs = tuple((int(item / factor) for item in img.size))
                    hrs = tuple((item * factor for item in lrs))
                    lr = np.asarray(img.resize(lrs))
                    hr = np.asarray(img.resize(hrs))
                    if len(lr.shape) == 3:
                        for left in range(0, lr.shape[2] - channel + 1, channel):
                            data_list.append(lr[:, :, left:left + channel])
                            label_list.append(hr[:, :, left:left + channel])
                    # elif len(lr.shape) == 2:
                    #     data_list.append(lr[:, :, None])
                    #     label_list.append(hr[:, :, None])
                elif resize:
                    hr = np.asarray(img.resize(hr_size))
                    lr = np.asarray(img.resize(lr_size))
                    data_list.append(lr)
                    label_list.append(hr)
                else:
                    stride = 3
                    lr_img_size = (round(img_width / factor), round(img_height / factor))
                    hr_img_size = (lr_img_size[0] * factor, lr_img_size[1] * factor)
                    lr_images = crop(img.resize(lr_img_size),
                                     lr_size[0], lr_size[1], stride)
                    hr_images = crop(img.resize(hr_img_size),
                                     hr_size[0], hr_size[1], stride * factor)
                    # print(len(lr_images), len(hr_images))
                    assert len(lr_images) == len(hr_images), "Length of LR images and HR images is not the same."
                    for lr, hr in zip(lr_images, hr_images):
                        lr = np.asarray(lr)
                        hr = np.asarray(hr)
                        if len(lr.shape) == 3:
                            for left in range(0, lr.shape[2] - channel + 1, channel):
                                data_list.append(lr[:, :, left:left + channel])
                                label_list.append(hr[:, :, left:left + channel])
                        elif len(lr.shape) == 2:
                            data_list.append(lr[:, :, None])
                            label_list.append(hr[:, :, None])
                    # for sub_img in crop(img, hr_size[0], hr_size[1], stride=14):
                    #     hr = (np.asarray(sub_img))
                    #     lr = (np.asarray(sub_img.resize(lr_size)))
                    #     if len(lr.shape) == 3:
                    #         for left in range(0, lr.shape[2] - channel + 1, channel):
                    #             data_list.append(lr[:, :, left:left + channel])
                    #             label_list.append(hr[:, :, left:left + channel])
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




