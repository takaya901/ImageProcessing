#%%
import cv2
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np
import glob
import math

def calc_abs(gt, gen, mask):
    error = 0
    count = 0
    for ch in range(3):
        for y in range(128):
            for x in range(128):
                if mask[y][x]:
                    count += 1
                    error += abs(gt[y][x][0] - gen[y][x][0])
                    print(x, y, error)
    return error / (255*3*count)

def compare_hist(gt, gen, mask):
    hist_size = 256
    ranges = (0, 256)
    errors = []

    for ch in range(3):
        gt_hist = cv2.calcHist([gt], [ch], mask, [hist_size], ranges)
        pred_hist = cv2.calcHist([gen], [ch], mask, [hist_size], ranges)
        errors.append(cv2.compareHist(gt_hist, pred_hist, 0))

    error = mean(errors)
    return (error + 1.) / 2.    #[-1,1]から[0,1]に変換

def rmse(gt, gen, mask):
    gt = gt.astype(np.int64)
    gen = gen.astype(np.int64)
    error = 0
    count = 0
# for ch in range(1):
    for y in range(128):
        for x in range(128):
            if mask[y][x]:
                # print(x, y)
                count += 1
                error += (gt[y][x] - gen[y][x]) ** 2
                # print(x, y, ch, (pred[y][x][ch] - gt[y][x][ch])**2)
                # print(error)

    # return error / (255**2 * count) #3 * 255**2 * (count / 3)
    return math.sqrt(error / count) if count != 0 else 0

test_path = 'C:/Users/bumpo/Documents/Research/dataset/RandomLight/gray/test/'
# test_path = '/Users/takaya/Documents/Research/Dataset/bunny/capsule/'
log_path = test_path + 'rmse.txt'

input_imgs = glob.glob(test_path + 'input/*.jpg')
img_num = len(input_imgs)
img_size = (128, 128)
error_list = []

# with open(log_path, mode='x') as f:
for i in range(img_num):
    # 入力画像とGroundTruthを128x128で読み込み
    input = cv2.imread(test_path + f'input/{str(i)}.jpg')
    input = cv2.resize(input, img_size)
    input_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    gt = cv2.imread(test_path + f'GT/{str(i)}.jpg')
    gt = cv2.resize(gt, img_size)
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    gen = cv2.imread(test_path + f'gen_100k/{str(i)}.jpg')
    gen = cv2.resize(gen, img_size)
    gen_gray = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY)

    # 入力とGTの差分マスクを作成
    # diff_color = cv2.absdiff(gt, pred)
    diff_gt = cv2.absdiff(input_gray, gt_gray)
    ret, mask_gt = cv2.threshold(diff_gt, 10, 255, cv2.THRESH_BINARY)
    # diff_gen = cv2.absdiff(input_gray, gen_gray)
    # ret, mask_gen = cv2.threshold(diff_gen, 10, 255, cv2.THRESH_BINARY)
    # mask = cv2.bitwise_or(mask_gt, mask_gen)
    mask = mask_gt
    # kernel = np.ones((5,5),np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.namedWindow('gt' + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow('gt' + str(i), gt)
    cv2.namedWindow('mask' + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow('mask' + str(i), mask)
    cv2.namedWindow('gen' + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow('gen' + str(i), input)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # error = rmse(gt_gray, gen_gray, mask)
    # if error != 0:
    #     error_list.append(error)
    # print(i, error)
    # f.write(f'{i} {error}\n')

print(f'mean:{mean(error_list)}')
print(f'stdev:{stdev(error_list)}')
    # f.write(f'mean:{mean(error_list)}\n')
    # f.write(f'stdev:{stdev(error_list)}')

# plt.hist(gt.ravel(),256,[0,256])
# plt.show()

# %%
