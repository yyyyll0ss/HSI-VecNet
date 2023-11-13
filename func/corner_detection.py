# encoding:utf-8
import cv2
import numpy as np
import os.path as osp
import os

filename = r'/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/pictures/indian_pines_gt.png'
save_path = '/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/corner_result'

img = cv2.imread(filename)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# 输入图像必须是float32， 最后一个参数[0.04,0.06]
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

img[dst > 0.1 * dst.max()] = [0, 0, 255]
cv2.imshow('img', img)
cv2.imshow('dst', dst)


file_path_ = osp.join(save_path, 'indian_pines')
if not osp.exists(file_path_):
    os.makedirs(file_path_)
file_path = osp.join(file_path_,'indian_pines_points_gt.png')
cv2.imwrite(file_path,img)