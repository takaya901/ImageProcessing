#%%
import cv2
import glob

path = 'C:/Users/bumpo/Documents/Research/dataset/white/'
color_path = path + 'color/'
gray_path = path + 'gray/'

imgs = glob.glob(color_path + 'train/input/*.jpg')
img_num = len(imgs)

for i in range(img_num):
    gray = cv2.imread(color_path + 'train/input/' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(gray_path + 'train/input/' + str(i) + '.jpg', gray)

imgs = glob.glob(color_path + 'train/GT/*.jpg')
img_num = len(imgs)

for i in range(img_num):
    gray = cv2.imread(color_path + 'train/GT/' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(gray_path + 'train/GT/' + str(i) + '.jpg', gray)

imgs = glob.glob(color_path + 'test/input/*.jpg')
img_num = len(imgs)

for i in range(img_num):
    gray = cv2.imread(color_path + 'test/input/' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(gray_path + 'test/input/' + str(i) + '.jpg', gray)    

imgs = glob.glob(color_path + 'test/GT/*.jpg')
img_num = len(imgs)

for i in range(img_num):
    gray = cv2.imread(color_path + 'test/GT/' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(gray_path + 'test/GT/' + str(i) + '.jpg', gray)

# imgs = glob.glob(data_path + '*.jpg')
# img_num = len(imgs)

# for i in range(img_num):
#     gray = cv2.imread(data_path + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
#     cv2.imwrite(out_path + str(i) + '.jpg', gray)

#%%
