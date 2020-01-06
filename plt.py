#coding:utf-8
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#画像の読み込み
# im = Image.open(r"C:\Users\bumpo\Documents\Research\dataset\RandomLight\gray\test\input\92.jpg")
im = Image.open(r"C:\Users\bumpo\Documents\Research\NDDS2\Captured\train\input\gray\0.jpg")
# im = Image.open(r"C:\Users\bumpo\Documents\Research\dataset\real_image\gray\2.jpg")

#画像をarrayに変換
im_list = np.asarray(im)
#貼り付け
plt.imshow(im_list)
#表示
plt.show()