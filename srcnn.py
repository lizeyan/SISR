import numpy as np


'''
这个文件负责处理所有的SRCNN问题数据的准备,结果的处理等琐碎的工作
'''


def load_data(width=28, height=28, factor=2, size=10):
    '''
    :param width: LR图片的宽度
    :param height: LR图片的高度
    :param factor: 从LR到HR的放大倍数,长和宽分别放大.
    :param size: 加载的图片数量
    :return: 图片的列表 train_data, test_data, train_label, test_label
    train_data和test_data的尺寸为[size,width, height, channel]
    train_label和test_label的尺寸为[size, width*factor, height*factor, channel]
    train和test数据不能有重叠
    '''
    hr_width = width * factor
    hr_height = height * factor
    train_data = np.random.rand(size, width, height, 1)
    test_data = np.random.rand(size, width, height, 1)
    train_label = np.random.rand(size, hr_width, hr_height, 1)
    test_label = np.random.rand(size, hr_width, hr_height, 1)
    return [train_data, test_data, train_label, test_label]
