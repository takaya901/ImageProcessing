#%%
import cv2
import glob

data_path = 'C:/Users/bumpo/Documents/Research/dataset/real/'
out_path = 'C:/Users/bumpo/Documents/Research/dataset/real/'
# out_path = data_path + 'gray/'
imgs = glob.glob(data_path + '*.jpg')
img_num = len(imgs)

for i in range(img_num):
    gray = cv2.imread(data_path + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(out_path + str(i) + '.jpg', gray)

#%%

data_path = 'C:/Users/bumpo/Documents/Research/dataset/RandomLight/color/'
out_path = 'C:/Users/bumpo/Documents/Research/dataset/RandomLight/gray/'
# out_path = data_path + 'gray/'
imgs = glob.glob(data_path + 'train/input/*.jpg')
img_num = len(imgs)

for i in range(img_num):
    gray = cv2.imread(data_path + 'train/input/' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(out_path + 'train/input/' + str(i) + '.jpg', gray)

imgs = glob.glob(data_path + 'train/GT/*.jpg')
img_num = len(imgs)

for i in range(img_num):
    gray = cv2.imread(data_path + 'train/GT/' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(out_path + 'train/GT/' + str(i) + '.jpg', gray)

imgs = glob.glob(data_path + 'test/input/*.jpg')
img_num = len(imgs)

for i in range(img_num):
    gray = cv2.imread(data_path + 'test/input/' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(out_path + 'test/input/' + str(i) + '.jpg', gray)    

imgs = glob.glob(data_path + 'test/GT/*.jpg')
img_num = len(imgs)

for i in range(img_num):
    gray = cv2.imread(data_path + 'test/GT/' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(out_path + 'test/GT/' + str(i) + '.jpg', gray)

imgs = glob.glob(data_path + '*.jpg')
img_num = len(imgs)

for i in range(img_num):
    gray = cv2.imread(data_path + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(out_path + str(i) + '.jpg', gray)

#%%
