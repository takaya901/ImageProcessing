#%%
import numpy as np
import glob
import h5py
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

dataset_path = 'C:/Users/bumpo/Documents/Research/dataset/white/gray'
hdf5_path =  'C:/Users/bumpo/Documents/Research/dataset/white/gray/hdf5'

orgs = []
masks = []
test_orgs = []
test_masks = []
target_size = (128, 128)

print('input target img')
files = glob.glob(dataset_path+'/train/input/*.jpg')
img_num = len(files)
for i in range(img_num):
    print(i)
    img = load_img(dataset_path+'/train/input/' + str(i) + '.jpg', target_size = target_size)
    imgarray = img_to_array(img)
    orgs.append(imgarray)
    
print('GT target img')
files = glob.glob(dataset_path+'/train/GT/*.jpg')
img_num = len(files)
for i in range(img_num):
    print(i)
    img = load_img(dataset_path+'/train/GT/' + str(i) + '.jpg', target_size = target_size)
    imgarray = img_to_array(img)
    masks.append(imgarray)

print('input target img')
files = glob.glob(dataset_path+'/test/input/*.jpg')
img_num = len(files)
for i in range(img_num):
    print(i)
    img = load_img(dataset_path+'/test/input/' + str(i) + '.jpg', target_size = target_size)
    imgarray = img_to_array(img)
    test_orgs.append(imgarray)

print('GT target img')
files = glob.glob(dataset_path+'/test/GT/*.jpg')
img_num = len(files)
for i in range(img_num):
    print(i)
    img = load_img(dataset_path+'/test/GT/' + str(i) + '.jpg', target_size = target_size)
    imgarray = img_to_array(img)
    test_masks.append(imgarray)

imgs = np.array(orgs)
gimgs = np.array(masks)
vimgs = np.array(test_orgs)
vgimgs= np.array(test_masks)
print('shapes')
print('org imgs  : ', imgs.shape)
print('mask imgs : ', gimgs.shape)
print('test org  : ', vimgs.shape)
print('test tset : ', vgimgs.shape)

outh5 = h5py.File(hdf5_path + '/dataset.hdf5', 'w')
outh5.create_dataset('TrainInputTarget', data=imgs)
outh5.create_dataset('TrainGTTarget', data=gimgs)
outh5.create_dataset('TestInputTarget', data=vimgs)
outh5.create_dataset('TestGTTarget', data=vgimgs)
outh5.flush()
outh5.close()

#%%
