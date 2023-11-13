# -*- coding: utf-8 -*-
"""
@author: YuZhu
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def color_map(dataset):
    # indian_pines
    indian_pines_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
                       [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
                       [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [238, 0, 0]])

    # salinas_valley
    """
    Brocoli_green_weed_1,Brocoli_green_weed_22,Fallow,Fallow_rough_plow,Fallow_smooth,Stubble,Celery,Grapes_untrained,
    Soil_vinyard_develop,Corn_senesced_green_weeds,Lettuce_romaine_4wk,Lettuce_romaine_5wk,Lettuce_romaine_6wk,Lettuce_romaine_7wk,
    Vinyard_untrained,Vinyard_vertical_trellis
    """
    salinas_valley_colors = np.array([[255, 255, 154], [0, 0, 255], [255, 48, 0], [0, 255, 154],
                       [255, 0, 255], [0, 48, 205], [47, 154, 255], [129, 129, 0],
                       [0, 255, 0], [154, 40, 154], [0, 154, 205], [102, 102, 154],
                       [147, 209, 79], [102, 48, 0], [0, 255, 255], [255, 255, 0]])
    # WHU_Hi_HongHu
    """
    Red_roof,Road,Bare_soil,Cotton,
    Cotton_firewood,Rape,Chinese_cabbage,Pakchoi,
    Cabbage,Tuber_mustard,Brassica_parachinensis,Brassica_chinensis,
    Small_Brassica_chinensis,Lactuca_sativa,Celtuce,Film_covered_lettuce,
    Romaine_lettuce,Carrot,White_radish,Garlic_sprout,
    Broad_bean,Tree
    """
    WHU_Hi_HongHu_colors = np.array([[255, 0, 0], [255, 255, 255], [176, 48, 96], [255, 255, 0],
                       [255, 127, 80], [0, 255, 0], [0, 205, 0], [0, 139, 0],
                       [127, 255, 212], [160, 32, 240], [216, 191, 216], [0, 0, 255],
                       [0, 0, 139], [218, 112, 214], [160, 82, 45], [0, 255, 255],
                       [255, 165, 0], [127, 255, 0], [139, 139, 0], [0, 139, 139],
                       [205, 181, 205],[238, 154, 0]])
    if dataset == 'salinas_valley':
        return salinas_valley_colors
    elif dataset == 'indian_pines':
        return indian_pines_colors
    elif dataset == 'WHU_Hi_HongHu':
        return WHU_Hi_HongHu_colors

def imgDraw(label, imgName, path='./pictures', show=True, dataset='indian_pines'):
    """
    功能：根据标签绘制RGB图
    输入：（标签数据，图片名）
    输出：RGB图
    备注：输入是2维数据，label有效范围[1,num]
    """
    row, col = label.shape
    numClass = int(label.max())
    Y_RGB = np.zeros((row, col, 3)).astype('uint8')  # 生成相同shape的零数组
    Y_RGB[np.where(label == 0)] = [0, 0, 0]  # 对背景设置为黑色
    colors = color_map(dataset)
    for i in range(1, numClass + 1):  # 对有标签的位置上色
        try:
            Y_RGB[np.where(label == i)] = colors[i - 1]
        except:
            Y_RGB[np.where(label == i)] = np.random.randint(0, 256, size=3)
    plt.axis("off")  # 不显示坐标
    if show:
        plt.imshow(Y_RGB)
    os.makedirs(path, exist_ok=True)
    plt.imsave(path + '/' + str(imgName) + '.png', Y_RGB)  # 分类结果图
    return Y_RGB




def imgDraw_mask(gt, imgName, path='./pictures'):
    """
    功能：根据图像分割的分块结果绘制边缘图
    输入：二维标签图
    输出：二维边缘图
    备注：
    """
    row, col = gt.shape
    gt_new = np.zeros((row + 2, col + 2))
    gt_new[1:-1, 1:-1] = gt + 1  # 将输入图补一圈零
    gt_edge = np.zeros(gt_new.shape)
    dif1 = gt_new[:-1, :] - gt_new[1:, :]  # 计算上下差异
    gt_edge[np.where(dif1 != 0)] = 1
    dif2 = gt_new[:, :-1] - gt_new[:, 1:]  # 计算左右差异
    gt_edge[np.where(dif2 != 0)] = 1
    dif3 = gt_new[1:, :] - gt_new[:-1, :]  # 计算上下差异
    dif3_ = np.zeros(gt_new.shape)
    dif3_[1:, :] = dif3
    gt_edge[np.where(dif3_ != 0)] = 1
    dif4 = gt_new[:, 1:] - gt_new[:, :-1]  # 计算左右差异
    dif4_ = np.zeros(gt_new.shape)
    dif4_[:, 1:] = dif4
    gt_edge[np.where(dif4_ != 0)] = 1
    gt_edge = gt_edge[1:-1, 1:-1]

    Y_RGB = np.zeros((row, col, 3)).astype('uint8')  # 生成相同shape的零数组
    Y_RGB[np.where(gt_edge == 0)] = [255, 255, 255]  # 背景设置为白色
    Y_RGB[np.where(gt_edge == 1)] = [255, 0, 0]  # 轮廓设置为红色
    os.makedirs(path, exist_ok=True)
    plt.imsave(path + '/' + str(imgName) + '.png', Y_RGB)  # 分类结果图
    return gt_edge


def imgDraw_error(out, gt, imgName, path='./pictures', show=True):
    """
    功能：绘制分类结果中错误的部分
    输入：模型输出结果,真值标签图
    输出：
    备注：输入均为2维,label有效范围[1,num]
    """
    row, col = gt.shape
    numClass = int(gt.max())
    Y_RGB = np.zeros((row, col, 3)).astype('uint8')  # 生成相同shape的零数组
    Y_RGB[np.where(gt == 0)] = [0, 0, 0]  # 对背景设置为黑色
    for i in range(1, numClass + 1):  # 对有标签的位置上色
        try:
            Y_RGB[np.where(out == i)] = colors[i - 1]
        except:
            Y_RGB[np.where(out == i)] = np.random.randint(0, 256, size=3)
        Y_RGB[np.where(gt == i)] = [0, 0, 0]  # 命中对的地方重置为黑色背景
    plt.axis("off")  # 不显示坐标
    if show:
        plt.imshow(Y_RGB)
    os.makedirs(path, exist_ok=True)
    plt.imsave(path + '/' + str(imgName) + '.png', Y_RGB)  # 分类结果图


def imgSaveRGB(img, imgName, path='./pictures'):
    """
    功能：保存RGB图
    输入：RGB图,保存名
    输出：
    备注：输入是2维数据，label范围随意
    """
    plt.axis("off")  # 不显示坐标
    plt.imshow(img)
    os.makedirs(path, exist_ok=True)
    plt.imsave(path + '/' + str(imgName) + '.png', img)  # 分类结果图


def imgDraw_idx(label, idx, imgName):
    """
    功能：根据索引绘制RGB图(绘制一部分)
    输入：（标签数据，图片名）
    输出：RGB图
    备注：输入是2维数据，idx是1维索引,label有效范围[1,num]
    """
    row, col = label.shape
    draw_arr = np.zeros((row * col))  # 绘制训练集与验证集
    draw_arr[idx] = (label.flatten())[idx]
    imgDraw(draw_arr.reshape(row, col).astype('uint8'), imgName=imgName)


def image_show(image):
    plt.axis('off')
    plt.imshow(image)
