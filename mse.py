import cv2
from statistics import mean, stdev
import matplotlib.pyplot as plt
import numpy as np

def calc_abs(gt, pred, mask):
    error = 0
    count = 0
    for ch in range(3):
        for y in range(128):
            for x in range(128):
                if mask[y][x]:
                    count += 1
                    error += abs(gt[y][x][0] - pred[y][x][0])
                    print(x, y, error)
    return error / (255*3*count)

def compare_hist(gt, pred, mask):
    hist_size = 256
    ranges = (0, 256)
    errors = []

    for ch in range(3):
        gt_hist = cv2.calcHist([gt], [ch], mask, [hist_size], ranges)
        pred_hist = cv2.calcHist([pred], [ch], mask, [hist_size], ranges)
        errors.append(cv2.compareHist(gt_hist, pred_hist, 0))

    error = mean(errors)
    return (error + 1.) / 2.    #[-1,1]から[0,1]に変換

def mse(gt, pred, mask):
    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)
    error = 0
    count = 0
    for ch in range(3):
        for y in range(128):
            for x in range(128):
                if mask[y][x]:
                    # print(x, y)
                    count += 1
                    error += (gt[y][x][ch] - pred[y][x][ch]) ** 2
                    # print(x, y, ch, (pred[y][x][ch] - gt[y][x][ch])**2)
                    # print(error)

    # print(count)
    return error / (255**2 * count) #3 * 255**2 * (count / 3)

test_path = 'C:/Users/bumpo/Documents/Research/dataset/RandomLight/gray/test/'
# test_path = '/Users/takaya/Documents/Research/Dataset/bunny/capsule/'
error_list = []

for i in range(1):
    #入力画像とGroundTruthを128x128で読み込み
    input = cv2.imread(test_path + 'without/{0}.jpg'.format(i))
    input = cv2.resize(input, (128, 128))
    input_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    gt = cv2.imread(test_path + 'with/{0}.jpg'.format(i))
    gt = cv2.resize(gt, (128, 128))
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    pred = cv2.imread(test_path + 'predict/{0}.jpg'.format(i))
    pred = cv2.resize(pred, (128, 128))
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

    #入力とGTの差分マスクを作成
    # diff_color = cv2.absdiff(gt, pred)
    diff_gt = cv2.absdiff(input_gray, gt_gray)
    ret, mask_gt = cv2.threshold(diff_gt, 10, 255, cv2.THRESH_BINARY)
    diff_pred = cv2.absdiff(input_gray, pred_gray)
    ret, mask_pred = cv2.threshold(diff_pred, 10, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask_gt, mask_pred)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.namedWindow('gt' + str(i), cv2.WINDOW_NORMAL)
    # cv2.imshow('gt' + str(i), mask_gt)
    # cv2.namedWindow('mask' + str(i), cv2.WINDOW_NORMAL)
    # cv2.imshow('mask' + str(i), mask)
    cv2.namedWindow('pred' + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow('pred' + str(i), mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#     error = compare_hist(gt, pred, mask)
#     error_list.append(error)
#     print(i, error)

# print('mean:{0}'.format(mean(error_list)))
# print('stdev:{0}'.format(stdev(error_list)))

# plt.hist(gt.ravel(),256,[0,256])
# plt.show()