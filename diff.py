# -*- coding: UTF-8 -*-
# https://www.blog.umentu.work/python3-opencv3で背景差分を求める/

#%%
import cv2
import numpy as np

path = 'C:/Users/bumpo/Documents/Research/dataset/real_image/'
input = cv2.imread(path + 'gray/0.jpg')
gen = cv2.imread(path + 'gen/0.jpg')

input = cv2.resize(input, (gen.shape[0], gen.shape[1]))
diff = cv2.absdiff(input, gen)

cv2.imshow('diff', diff)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
