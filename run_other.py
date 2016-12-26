from srcnn import load_data
from PIL import Image
import numpy as np
from utils import log
import sys

def evaluation_PSNR(data, label):
    mse = np.mean((data - label) ** 2)
    return 20 * np.log10(255) - 10 * np.log10(mse)


class Test(object):
    def __init__(self, factor=2):
        self.factor = factor
        self.test_data = []
        self.test_label = []

    def load_test_data(self):
        self.test_data, self.test_label = load_data(["./data/Test/Set5"], 128, 128, factor=self.factor, channel=3, size=100)

    def test_bicubic(self):
        res = []
        for i in range(len(self.test_data)):
            hr_image = Image.fromarray(self.test_label[i])
            lr_image = Image.fromarray(self.test_data[i])
            hr_pdt = lr_image.resize(hr_image.size, Image.BICUBIC)
            psnr = evaluation_PSNR(np.asarray(hr_pdt), np.asarray(hr_image))
            res.append(psnr)
            lr_image.save("./test_results/%d_input.jpg" % (i+1))
            hr_image.save("./test_results/%d_label.jpg" % (i+1))
            hr_pdt.save("./test_results/%d_bicubic_%f.jpg" % (i+1, psnr))
        return np.mean(res)

    def test_bilinear(self):
        res = []
        for i in range(len(self.test_data)):
            hr_image = Image.fromarray(self.test_label[i])
            lr_image = Image.fromarray(self.test_data[i])
            hr_pdt = lr_image.resize(hr_image.size, Image.BILINEAR)
            res.append(evaluation_PSNR(np.asarray(hr_pdt), np.asarray(hr_image)))
            lr_image.save("./test_results/%d_input.jpg" % (i+1))
            hr_image.save("./test_results/%d_label.jpg" % (i+1))
            hr_pdt.save("./test_results/%d_bilinear.jpg" % (i+1))
        return np.mean(res)

    def test_antialias(self):
        res = []
        for i in range(len(self.test_data)):
            hr_image = Image.fromarray(self.test_label[i])
            lr_image = Image.fromarray(self.test_data[i])
            hr_pdt = lr_image.resize(hr_image.size, Image.ANTIALIAS)
            res.append(evaluation_PSNR(np.asarray(hr_pdt), np.asarray(hr_image)))
            lr_image.save("./test_results/%d_input.jpg" % (i+1))
            hr_image.save("./test_results/%d_label.jpg" % (i+1))
            hr_pdt.save("./test_results/%d_antialias.jpg" % (i+1))
        return np.mean(res)

if len(sys.argv) > 1:
    test = Test(factor=eval(sys.argv[1]))
else:
    test = Test()
test.load_test_data()
result1 = test.test_bicubic()
result2 = test.test_bilinear()
result3 = test.test_antialias()
log("test PSNR of bicubic is %f" % result1)
log("test PSNR of bilinear is %f" % result2)
log("test PSNR of antialias is %f" % result3)
