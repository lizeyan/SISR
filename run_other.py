from srcnn import load_data
from PIL import Image
from solve_srcnn import evaluation_PSNR
import numpy as np
from utils import log

class Test(object):
    def __init__(self):
        self.factor = 2
        self.test_data = []
        self.test_label = []

    def load_test_data(self):
        self.test_data, self.test_label = load_data(["./data/test"], 512, 512, factor=self.factor, size=10)

    def test_bicubic(self):
        res = []
        for i in range(len(self.test_data)):
            hr_image = Image.fromarray(self.test_label[i])
            lr_image = Image.fromarray(self.test_data[i])
            hr_pdt = lr_image.resize((1024, 1024), Image.BICUBIC)
            res.append(evaluation_PSNR(np.asarray(hr_pdt), np.asarray(hr_image)))
            lr_image.save("./test_results/%d_input.jpg" % (i+1))
            hr_image.save("./test_results/%d_label.jpg" % (i+1))
            hr_pdt.save("./test_results/%d_bicubic.jpg" % (i+1))
        return np.mean(res)

    def test_bilinear(self):
        res = []
        for i in range(len(self.test_data)):
            hr_image = Image.fromarray(self.test_label[i])
            lr_image = Image.fromarray(self.test_data[i])
            hr_pdt = lr_image.resize((1024, 1024), Image.BILINEAR)
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
            hr_pdt = lr_image.resize((1024, 1024), Image.ANTIALIAS)
            res.append(evaluation_PSNR(np.asarray(hr_pdt), np.asarray(hr_image)))
            lr_image.save("./test_results/%d_input.jpg" % (i+1))
            hr_image.save("./test_results/%d_label.jpg" % (i+1))
            hr_pdt.save("./test_results/%d_antialias.jpg" % (i+1))
        return np.mean(res)

test = Test()
test.load_test_data()
result1 = test.test_bicubic()
result2 = test.test_bilinear()
result3 = test.test_antialias()
log("test PSNR of bicubic is %f" % result1)
log("test PSNR of bilinear is %f" % result2)
log("test PSNR of antialias is %f" % result2)
