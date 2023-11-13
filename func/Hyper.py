# -*- coding: utf-8 -*-
"""
@author: YuZhu
"""
import numpy as np
import cv2

# plt.switch_backend('agg')
# sys.path.append("..")


def calcAccuracy(predict, label):
    """
    功能：计算predict相当于label的正确率
    输入：（预测值，真实值）
    输出：正确率
    备注：输入均为一维数据，此时标签已经减一处理
    """
    num = len(label)
    numClass = np.max(label) + 1
    # print(len(predict))
    # print(len(label))

    OA = np.sum(predict == label) * 1.0 / num
    correct_sum = np.zeros(numClass)
    reali = np.zeros(numClass)
    predicti = np.zeros(numClass)
    producerA = np.zeros(numClass)
    for i in range(0, numClass):  # 对于每个类别 0-8
        correct_sum[i] = np.sum(label[np.where(predict == i)] == i)  # 该类别预测对的数目
        reali[i] = np.sum(label == i)  # 该类别真正的数目
        predicti[i] = np.sum(predict == i)  # 该类别预测的数目
        producerA[i] = correct_sum[i] / reali[i]  # 该类别的正确率
    Kappa = (num * np.sum(correct_sum) - np.sum(reali * predicti)) * 1.0 / (
            num * num - np.sum(reali * predicti))  # 计算出某综合参数
    print('OA:', OA)
    print('AA:', producerA.mean(), 'for each part:', producerA)
    print('Kappa:', Kappa)
    # for i in producerA:
    #     print("%.3f" % i)
    # print("%.3f" % OA)
    # print("%.3f" % producerA.mean())
    # print("%.3f" % Kappa)
    return OA, Kappa, producerA


def create_patch(zeroPaddedX, row_index, col_index, patchSize=5):
    """
    功能：对指定像素切割一次patch
    输入：（零增广数据，行坐标，列坐标，patch大小）
    输出：该指定像素以及周围数据所构成的patch
    备注：返回如5*5*103型数据
    """
    row_slice = slice(row_index, row_index + patchSize)
    col_slice = slice(col_index, col_index + patchSize)
    patch = zeroPaddedX[row_slice, col_slice, :]
    return np.asarray(patch).astype(np.float32)


def dataEnrich(X, y):
    """
    功能：对数据扩充增强
    输入：（原始数据X，标签数据y）
    输出：（扩充后的数据X，扩充后的标签数据y）
    备注：随机对patch旋转角度而更充分地利用数据,5*5*103
    """
    from scipy.ndimage.interpolation import rotate
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts
    # repeat for every label and concat
    newX = X[y == uniqueLabels[0], :, :, :].repeat(round(labelInverseRatios[0]), axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y == label, :, :, :].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :, :, :]
    newY = newY[rand_perm]
    # random flip each patch
    for i in range(int(newX.shape[0] / 2)):
        patch = newX[i, :, :, :]
        num = np.random.randint(0, 3)
        if num == 0:
            flipped_patch = np.flipud(patch)  # 矩阵上下翻转函数
        if num == 1:
            flipped_patch = np.fliplr(patch)  # 矩阵左右翻转函数
        if num == 2:
            no = (np.random.randint(12) - 6) * 30
            flipped_patch = rotate(patch, no, axes=(1, 0),  # 矩阵旋转函数
                                   reshape=False, output=None, order=3, mode='constant',
                                   cval=0.0, prefilter=False)
        newX[i, :, :, :] = flipped_patch  # 替换随机翻转后的数据
    return newX, newY


def dataMess(X_train, y_train):
    """
    功能：按照随机顺序打乱两个patch序列
    输入：（原始数据X，原始标签y）
    输出：打乱后的两个序列
    备注：无
    """
    x_trains_alea_indexs = list(range(len(X_train)))  # 生成一串序列
    np.random.shuffle(x_trains_alea_indexs)  # 对该序列进行乱序
    X_train = X_train[x_trains_alea_indexs]  # 根据该序列打乱训练序列
    y_train = y_train[x_trains_alea_indexs]  # 根据该序列打乱标记值
    return X_train, y_train


def dataNormalize(X, type=1):
    """
    功能：数据归一化
    输入：（原始数据，归一化类型）
    输出：归一化后数据
    备注：type==1最常用;与PCA二选一
        type==0 x = (x-mean)/std(x) #标准化
        type==1 x = (x-min(x))/(max(x)-min(x)) #归一化
        type==2 x = (2x-max(x))/(max(x))
    """
    if type == 0:  # 均值0，max、min不定
        mu = np.mean(X)
        X_norm = X - mu
        sigma = np.std(X_norm)
        X_norm = X_norm / sigma
        return X_norm
    elif type == 1:  # [0,1],最常用
        minX = np.min(X)
        maxX = np.max(X)
        X_norm = X - minX
        X_norm = X_norm / (maxX - minX)
    elif type == 2:  # 均值非零，[-1,1]
        maxX = np.max(X)
        X_norm = 2 * X - maxX
        X_norm = X_norm / maxX
    return X_norm.astype(np.float32)


def displayClassTable(n_list, matTitle=""):
    """
    功能：打印list的各元素
    输入：（list）
    输出：无
    备注：无
    """
    from pandas import DataFrame
    print("\n+--------- 原始输入数据" + matTitle + "统计结果 ------------+")
    lenth = len(n_list)  # 一共n个分类
    column = range(1, lenth + 1)
    table = {'Class': column, 'Total': [int(i) for i in n_list]}
    table_df = DataFrame(table).to_string(index=False)
    print(table_df)
    print('All available data total ' + str(int(sum(n_list))))
    print("+---------------------------------------------------+")


def felzenszwalb_hsi(data_hsi, min_size=25, scale=1, sigma=0.8):
# def felzenszwalb_hsi(data_hsi, min_size=25, scale=0.01, sigma=0.01):
    # seg = felzenszwalb_hsi(image, min_size=1000);
    # img = label2rgb(seg);
    # imgSaveRGB(img, 'test.png')

    from skimage.segmentation import felzenszwalb
    return felzenszwalb(data_hsi, scale=scale, sigma=sigma, min_size=min_size)


def generate_iter(data, gt, train_indices, test_indices, total_indices, patchSize=9, batchSize=128):
    """
    功能：对patch生成数据迭代器
    输入：原始数据X,原始数据y,训练集坐标,测试集坐标,总标注坐标,patchSize,batchSize
    输出：训练集迭代器\测试集迭代器\总坐标迭代器
    备注：原始数据均为2D,坐标均为1D
    """
    import torch
    import torch.utils.data as Data
    all_data = patchesCreate_all(data, patchSize).transpose(0, 3, 1, 2)
    gt = gt.flatten()

    train_data = all_data[train_indices]
    test_data = all_data[test_indices]
    all_data = all_data[total_indices]
    train_gt = gt[train_indices] - 1
    test_gt = gt[test_indices] - 1
    all_gt = gt[total_indices] - 1

    x1_tensor_train = torch.from_numpy(train_data)
    y1_tensor_train = torch.from_numpy(train_gt).type(torch.LongTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    x1_tensor_test = torch.from_numpy(test_data)
    y1_tensor_test = torch.from_numpy(test_gt).type(torch.LongTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, y1_tensor_test)

    all_tensor_data = torch.from_numpy(all_data)
    all_tensor_data_label = torch.from_numpy(all_gt).type(torch.LongTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=batchSize,
        shuffle=True,
        num_workers=4,
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=batchSize,
        shuffle=False,
        num_workers=4,
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,
        batch_size=batchSize,
        shuffle=False,
        num_workers=4,
    )
    return train_iter, test_iter, all_iter


def imageCRF(CNNMap, prob):
    """
    功能：对卷积网络结果进行CRF优化
    输入：(倒数第二层卷积结果,最后一层卷积结果再softmax)
    输出：新的标签数据
    备注：无
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
    row, col, layers = CNNMap.shape
    numClass = prob.shape[2]
    # CNNMap = dataNormalize(PCANorm(CNNMap, numPCA).reshape(row * col, numPCA)).reshape(row, col, numPCA)# 优化网络数据
    CNNMap = dataNormalize(PCANorm(CNNMap, numPCA=5))  # 优化网络数据，需在[0,1]
    softmax = prob.transpose((2, 0, 1))  # 9*610*340 卷积计算的结果
    unary = unary_from_softmax(softmax)  # 转换为一元势，实为取-log(softmax)
    unary = np.ascontiguousarray(unary)  # 将内存不连续存储的数组转换为连续的，使运行更快
    d = dcrf.DenseCRF(row * col, numClass)
    d.setUnaryEnergy(unary)  # 将一元势添加进CRF中
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=(row, col))  # 创建二元高斯势，sdim每个维度的比例因子
    d.addPairwiseEnergy(feats, compat=1, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)  # 添加进CRF
    feats = create_pairwise_bilateral(sdims=(30, 30), schan=[5], img=CNNMap, chdim=2)
    # 创建二元双边势，每个维度比例因子，通道的比例因子，原数据，通道位于原数据的维度
    d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)  # 添加进CRF
    Q = d.inference(5)  # 将一元势和二元势结合起来就能够比较全面地去考量像素之间的关系，并得出优化后的结果
    new_label = np.argmax(Q, axis=0).reshape((row, col))
    return new_label


def kMeans(dataSet, k):
    """
    函数说明：k-mean聚类算法
    Parameter：
        dataSet：数据集，格式如 10000*2
        k:分类类别数
    Return：
        k个质心坐标（格式如10*2）、样本的分配结果（格式如10000*1）
    """
    from sklearn.cluster import KMeans
    labels = KMeans(k, init='k-means++', n_init=k, max_iter=3000).fit(dataSet).labels_
    # KMeans(16,init='k-means++',n_init=16,max_iter=3000).fit(test_gt_pred).labels_
    return labels


def listClassification(Y, matTitle=''):
    """
    功能：对标签数据计数并打印
    输入：（原始标签数据，是否打印）
    输出：分类结果
    备注：无
    """
    numClass = np.max(Y)  # 获取分类数
    listClass = []  # 用列表依次存储各类别的数量
    for i in range(numClass):
        listClass.append(len(np.where(Y == (i + 1))[0]))
    displayClassTable(listClass, matTitle)
    return listClass

# 彩色图像全局直方图均衡化
def hisEqulColor1(img):
	# 将RGB图像转换到YCrCb空间中
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # 将YCrCb图像通道分离
    channels = cv2.split(ycrcb)
    # 对第1个通道即亮度通道进行全局直方图均衡化并保存
    cv2.equalizeHist(channels[0],channels[0])
    # 将处理后的通道和没有处理的两个通道合并，命名为ycrcb
    cv2.merge(channels,ycrcb)
    # 将YCrCb图像转换回RGB图像
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


# 彩色图像进行自适应直方图均衡化，代码同上的地方不再添加注释
def hisEqulColor2(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)

    # 以下代码详细注释见官网：
    # https://docs.opencv.org/4.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])

    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def matDataLoad(filename='./datasets/PaviaU_gt.mat'):
    """
    功能：读取.mat文件的有效数据
    输入：（文件名字）
    输出：数据
    备注：无
    """
    try:  # 读取v5类型mat数据
        import scipy.io
        data = scipy.io.loadmat(filename)
        x = list(data.keys())
        data = data[x[-1]]  # 读取出有用数据
    except:  # 读取v7类型mat数据
        import h5py
        data = h5py.File(filename, 'r')
        x = list(data.keys())
        data = data[x[1]][()].transpose(2, 1, 0)  # 调整为（row,col,band）
    if 'gt' in filename:
        return data.astype('long')  # 标签数据输出为整型
    else:
        return data.astype('float32')  # 原始数据输出为浮点型

def get_dataset_rgb(dataset):
    """
    get the rgb picture of hsi dataset
    """
    import skimage.io as io
    import cv2
    import numpy as np
    if dataset == 'indian_pines':
        data_hsi = matDataLoad(f'../datasets/{dataset}/' + dataset + '.mat')
        rgb_data = np.array(data_hsi)[:,:,[29,19,9]]
        io.imsave(f'../{dataset}_rgb.png',rgb_data)
    if dataset == 'salinas_valley':
        data_hsi = matDataLoad(f'../datasets/{dataset}/' + dataset + '.mat')
        rgb_data = np.array(data_hsi)[:,:,[29,19,9]]
        io.imsave(f'../{dataset}_rgb.png',rgb_data)
    if dataset == 'WHU_Hi_HongHu':
        data_hsi = matDataLoad(f'../datasets/{dataset}/' + dataset + '.mat')
        rgb_data = np.array(data_hsi)[:,:,[100,66,29]]
        io.imsave(f'../{dataset}_rgb.png',rgb_data)

    if dataset == 'WHU_Hi_LongKou':
        data_hsi = matDataLoad(f'../datasets/{dataset}/' + dataset + '.mat')
        rgb_data = np.array(data_hsi)[:,:,[100,66,29]]
        io.imsave(f'../{dataset}_rgb.png',rgb_data)
        #his
        img = cv2.imread(f'../{dataset}_rgb.png')
        img2 = img.copy()
        res2 = hisEqulColor2(img2)
        cv2.imwrite(f'../{dataset}_rgb.png',res2)
        #enhance lightness
        img = cv2.imread(f'../{dataset}_rgb.png')
        img_t = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_t)
        v1 = np.clip(cv2.add(1*v,30),0,255)
        img1 = np.uint8(cv2.merge((h,s,v1)))
        img1 = cv2.cvtColor(img1,cv2.COLOR_HSV2BGR)
        cv2.imwrite(f'../{dataset}_rgb.png',img1)

    if dataset == 'PaviaU':
        data_hsi = matDataLoad(f'../datasets/{dataset}/' + dataset + '.mat')
        rgb_data = np.array(data_hsi)[:,:,[20,40,60]]
        io.imsave(f'../{dataset}_rgb.png',rgb_data)

        img = cv2.imread(f'../{dataset}_rgb.png')
        img2 = img.copy()
        res2 = hisEqulColor2(img2)
        cv2.imwrite(f'../{dataset}_rgb.png',res2)

    if dataset == 'Houston':
        data_hsi = matDataLoad(f'../datasets/{dataset}/' + dataset + '.mat')
        rgb_data = np.array(data_hsi)[:,:,[20,50,80]]
        io.imsave(f'../{dataset}_rgb.png',rgb_data)

        img = cv2.imread(f'../{dataset}_rgb.png')
        img2 = img.copy()
        res2 = hisEqulColor2(img2)
        cv2.imwrite(f'../{dataset}_rgb.png',res2)

#get_dataset_rgb('PaviaU')


def pointsLoad(json_path):
    """
    load the points json file ,return a map of points
    """
    import json
    import torch
    with open(json_path,'r')as f:
        data = json.load(f)
        instances = data['annotations']
        junctions = [i for instance in instances  for i in instance['segmentation'][0]]
        junctions = torch.tensor([[junctions[i],junctions[i+1]] for i in range(0,len(junctions),2)])
        height, width = data['images'][0]['height'], data['images'][0]['width']
        jmap = torch.zeros((height, width), dtype=torch.long)
        joff = torch.zeros((2, height, width), dtype=torch.float32)

        xint, yint = junctions[:,0].long(), junctions[:,1].long()
        off_x = junctions[:,0] - xint.float()-0.5
        off_y = junctions[:,1] - yint.float()-0.5
        jmap[yint, xint] = 1
        joff[0, yint, xint] = off_x
        joff[1, yint, xint] = off_y

        target = {
            'jloc': jmap[None],
            'joff': joff,
        }
        return target



def padWithZeros(X, patchSize=5):
    """
    功能：对数据进行零增广
    输入：（原始数据，patch大小）
    输出：零增广后数据
    备注：无
    """
    zeroSize = int((patchSize - 1) / 2)  # 零增广个数
    zeroPaddedX = np.zeros((X.shape[0] + 2 * zeroSize, X.shape[1] + 2 * zeroSize, X.shape[2]))
    x_offset = zeroSize
    y_offset = zeroSize
    zeroPaddedX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return zeroPaddedX.astype(np.float32)


def patchesCreate(X, y, patchSize=9):
    """
    功能：将数据挨像素切割成patch
    输入：（原始数据X，原始标签数据y，patch大小）
    输出：分patch之后的数据与标签数据并打印
    备注：分割前是“块”，分割后是“条”
    """
    numClass = np.max(y)
    row, col = y.shape
    zeroPaddedX = padWithZeros(X, patchSize)
    nb_samples = np.zeros(numClass)  # 用于统计每个类别的数据
    patchesData = []  # 用于收集有效像素的patch
    patchesLabels = []  # 收集patch对应的标签
    for i in range(row):
        for j in range(col):  # 对于610*340这么多每个像素的1*103数据
            label = y[i, j]  # 提取对应像素的标注值，若为零则为背景，应该剔除掉
            if label > 0:  # 仅保留有用标注
                patch = create_patch(zeroPaddedX, i, j, patchSize)  # 5*5*103，以该像素为中心切取周围两圈构成5*5
                nb_samples[label - 1] += 1  # 统计每种类型各自的数量
                patchesData.append(patch.astype(np.float32))  # 保存每种类型的数据
                patchesLabels.append(label)  # 保存该patch的标签
    displayClassTable(nb_samples)  # 打印每种类型对应的数量
    return np.array(patchesData), np.array(patchesLabels)


def patchesCreate_all(X, patchSize=9):
    """
    功能：将数据挨像素切割成patch(包括背景部分)
    输入：（原始数据X，patch大小）
    输出：分patch之后的数据
    备注：不输入标签数据
    """
    row, col, layers = X.shape
    zeroPaddedX = padWithZeros(X, patchSize)
    patchesData = []  # 用于收集有效像素的patch
    for i in range(row):
        for j in range(col):  # 对于610*340这么多每个像素的1*103数据
            patch = create_patch(zeroPaddedX, i, j, patchSize)  # 5*5*103，以该像素为中心切取周围两圈构成5*5
            patchesData.append(patch.astype(np.float32))  # 保存每种类型的数据
    return np.array(patchesData).astype(np.float32)


def patchesCreate_balance(X, y, patchSize=5, train_mode=False):
    """
    功能：重构二分类高光谱数据集,使两者数量平衡
    输入：（原始数据X，原始数据y,patch大小,是否为训练模式）
    输出：分patch之后的数据及其标签
    备注：若为测试模式则全部取出,不进行平衡
    """
    row, col = y.shape
    zeroPaddedX = padWithZeros(X, patchSize)
    nb_samples = np.zeros(2)  # 用于统计每个类别的数据
    patchesData = []  # 用于收集有效像素的patch
    patchesLabels = []  # 收集patch对应的标签
    if train_mode:
        random_choose = np.random.randint(15, size=(row, col))
    else:
        random_choose = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            label = y[i, j]
            if label > 0 or random_choose[i, j] == 0:  # 由于背景过多,故降低背景样本数量
                patch = create_patch(zeroPaddedX, i, j, patchSize)
                nb_samples[label] += 1  # 统计每种类型各自的数量
                patchesData.append(patch.astype(np.float32))  # 保存每种类型的数据
                patchesLabels.append(label)  # 保存该patch的标签
    displayClassTable(nb_samples)  # 打印每种类型对应的数量
    return np.ascontiguousarray(patchesData).astype('float32'), np.ascontiguousarray(patchesLabels).astype('long')


def PCANorm(X, numPCA=3):
    """
    功能：PCA降维
    输入：（原始数据，降维后维度）
    输出：降维后数据
    备注：输入输出都为三维数据
    """
    from sklearn.decomposition import PCA  # PCA降维
    newX = np.reshape(X, (-1, X.shape[2]))  # 将空间信息铺开
    pca = PCA(n_components=numPCA, whiten=True)  # 定义PCA信息
    newX = pca.fit_transform(newX)  # 降维操作
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numPCA))  # 整形回原来的空间形状
    return newX


def print_f(data='', fileName='data', show=True):
    """
    功能：对数据既打印又保存到.txt文件到本地
    输入：（打印数据，保存为的文件名）
    输出：无
    备注：无
    """
    import os
    root_path = '../records/'
    os.makedirs(root_path, exist_ok=True)
    with open(root_path + fileName + '.txt', 'a') as f:  # 'a'表示append,即在原来文件内容后继续写数据（不清除原有数据）
        f.write(str(data) + "\n")
    if show:
        print(data)


def one_hot_encoding(Y):
    """
    功能：对标签数据进行one-hot编码
    输入：1维标签数据
    输出：2维编码后数据
    备注：处理Y的标签 例如（光谱维度9,数据2） --> [ 0, 0, 1, 0, 0, 0, 0, 0, 0],此时标签没有减一处理
    """
    numClass = np.max(Y)
    y_encoded = np.zeros((Y.shape + tuple([numClass])), 'uint8')
    for i in range(1, numClass + 1):
        index = np.where(Y == i)
        if len(Y.shape) == 1:  # 如果是一维数据
            y_encoded[index[0], i - 1] = 1
        else:  # 如果是二维数据
            y_encoded[index[0], index[1], i - 1] = 1
    return y_encoded


def splitTrainTestSet(X, y, testRatio=0.90):
    """
    功能：按比例分割训练集与测试集
    输入：（原始数据X，原始标签y，验证集占比）
    输出：分割后的数据集
    备注：无
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=345, stratify=y)
    print("\n+---------------- 训练集测试集数据概览 ---------------------+")
    print('x_train.shape: ' + str(X_train.shape))
    print('y_train.shape: ' + str(y_train.shape))
    print('x_test.shape: ' + str(X_test.shape))
    print('y_test.shape: ' + str(y_test.shape))
    print("+---------------------------------------------------+")
    return X_train, X_test, y_train, y_test


def split_dataset(gt_1D, gt_jloc_1D, numTrain=0.3, is_mask=True):
    """
    功能：按照每类数量分割训练集与测试集
    输入：（1-d class label ，1-d jloc label, 每类训练数量or proportion）
    输出：训练集一维坐标,测试集一维坐标
    备注：当某类别数量过少时,就训练集测试集复用
    """
    np.random.seed(0)
    if is_mask:
        train_idx, test_idx, numList = [], [], []
        numClass = np.max(gt_1D)  # 获取最大类别数
        for i in range(0, numClass + 1):  # 忽略背景元素start from 1,if not from 0
            if i == 0:
                class_idx = np.where(gt_1D == i)[0]  # 记录下该类别的坐标值
                Train = int(len(class_idx) * numTrain)

                np.random.shuffle(class_idx)  # 对坐标乱序
                train_idx.append(class_idx[:Train])
                test_idx.append(class_idx[Train:])
                numList.append(len(class_idx[:Train]))
            else:
                class_idx = np.where(gt_1D == i)[0]  # 记录下该类别的坐标值
                all_jloc_idx = np.where(gt_jloc_1D == 1)[0]  # save the idx of junction

                # per class jloc positive samples
                jloc_idx = np.intersect1d(class_idx, all_jloc_idx)

                Train = int(len(class_idx) * numTrain)
                jloc_Train = int(len(jloc_idx) * 1)

                if len(jloc_idx) > 0:  # unlabel don't have positive sample
                    np.random.shuffle(jloc_idx)  # 对坐标乱序
                    train_idx.append(jloc_idx[:jloc_Train])  # get every class positive samples

                # per class jloc negetive samples
                no_jloc_idx = np.setdiff1d(class_idx, jloc_idx)

                np.random.shuffle(no_jloc_idx)  # 对坐标乱序
                train_idx.append(no_jloc_idx[:(Train - jloc_Train)])  # get every class negetive samples

                test_idx.append(jloc_idx[jloc_Train:])  # 收集每一类的测试坐标
                test_idx.append(no_jloc_idx[(Train - jloc_Train):])

        train_idx = np.asarray([item for sublist in train_idx for item in sublist])
        test_idx = np.asarray([item for sublist in test_idx for item in sublist])
        return train_idx, test_idx, numList
    else:
        train_idx, test_idx, numList = [], [], []
        numClass = np.max(gt_1D)  # 获取最大类别数

        for i in range(0,numClass + 1):
            class_idx = np.where(gt_1D == i)[0]  # 记录下该类别的坐标值
            all_jloc_idx = np.where(gt_jloc_1D == 1)[0]  # save the idx of junction

            # per class jloc positive samples
            jloc_idx = np.intersect1d(class_idx,all_jloc_idx)

            Train = int(len(class_idx) * 0.4)
            jloc_Train = int(len(jloc_idx) * 1)   # 每类节点数量太少，因此在训练及测试集中复用

            if len(jloc_idx) > 0:   # unlabel don't have positive sample
                np.random.shuffle(jloc_idx)  # 对坐标乱序
                train_idx.append(jloc_idx[:jloc_Train])  # get every class positive samples

            # per class jloc negetive samples
            no_jloc_idx = np.setdiff1d(class_idx,jloc_idx)

            np.random.shuffle(no_jloc_idx)  # 对坐标乱序
            train_idx.append(no_jloc_idx[:(Train-jloc_Train)])  # get every class negetive samples

            #test_idx.append(jloc_idx[jloc_Train:])  # 收集每一类的测试坐标
            test_idx.append(jloc_idx[-(jloc_Train):])  # 收集每一类的测试坐标
            test_idx.append(no_jloc_idx[(Train - jloc_Train):])

        train_idx = np.asarray([item for sublist in train_idx for item in sublist])
        test_idx = np.asarray([item for sublist in test_idx for item in sublist])

        return train_idx, test_idx, numList



def us_calcAccuracy(pred_targets, targets, numClass=9):
    """
    功能：计算无监督predict相当于label的正确率
    输入：（预测值，真实值）
    输出：正确率
    备注：输入均为一维数据，在程序内部自动处理了“预测值是相对值”的问题
    """
    numTotal = len(pred_targets)  # 总数
    for i in range(numClass):  # 对错位进行调换
        idx = np.where(targets == i)
        freq_most = np.argmax(np.bincount(pred_targets[idx]))
        # _, hist = np.unique(pred_targets[idx], return_counts=True)
        # freq_most = np.argmax(hist)  # 计算众数
        idx_ = np.where(pred_targets == i)
        if i != freq_most:
            pred_targets[idx] = i
            pred_targets[idx_] = freq_most
    OA = np.sum(pred_targets == targets) * 1.0 / numTotal
    return OA


def color_map(dataset):
    # indian_pines
    indian_pines_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                                    [0, 255, 255], [255, 0, 255],[176, 48, 96], [46, 139, 87],
                                    [160, 32, 240], [255, 127, 80],[127, 255, 212],[218, 112, 214],
                                    [160, 82, 45], [127, 255, 0], [216, 191, 216], [238, 0, 0]])

    # salinas_valley
    """
    Brocoli_green_weed_1,Brocoli_green_weed_22,Fallow,Fallow_rough_plow,
    Fallow_smooth,Stubble,Celery,Grapes_untrained,
    Soil_vinyard_develop,Corn_senesced_green_weeds,Lettuce_romaine_4wk,Lettuce_romaine_5wk,
    Lettuce_romaine_6wk,Lettuce_romaine_7wk,Vinyard_untrained,Vinyard_vertical_trellis
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
    WHU_Hi_LongKou_colors = np.array([[255, 0, 0], [235, 154, 0], [255, 255, 0], [0, 255, 0],
                       [0, 255, 255], [0, 139, 139], [0, 0, 255], [255, 255, 255],
                       [160, 32, 240]])
    # Houston
    Houston_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
                               [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
                               [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [238, 0, 0]])
    # PaviaU
    PaviaU_colors = np.array([[191, 191, 191], [0, 255, 0], [0, 255, 255], [0, 128, 0], [255, 0, 255], [164, 81, 40],
                              [128, 0, 128], [255, 0, 255], [255, 255, 0]])

    if dataset == 'salinas_valley':
        return salinas_valley_colors
    elif dataset == 'indian_pines':
        return indian_pines_colors
    elif dataset == 'WHU_Hi_HongHu':
        return WHU_Hi_HongHu_colors
    elif dataset == 'WHU_Hi_LongKou':
        return WHU_Hi_LongKou_colors
    elif dataset == 'PaviaU':
        return PaviaU_colors
    elif dataset == 'Houston':
        return Houston_colors

def draw_poly_result(all_class_polygon,save_path,dataset,numTrain):
    import cv2
    import os.path as osp
    import os
    # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
    #           [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
    #           [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [238, 0, 0]]
    colors = color_map(dataset).tolist()
    result = cv2.imread(f'/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/{dataset}_rgb.png')
    #result = np.zeros((145, 145, 3)).astype('uint8')
    for key,values in all_class_polygon.items():
        for polygon in values:
            polygon = np.array(polygon,np.int32)
            #print(polygon)
            cv2.polylines(result,[polygon],True,colors[int(key)-1][::-1],1)

    file_path = osp.join(save_path,dataset)
    if not osp.exists(file_path):
        os.makedirs(file_path)

    cv2.imwrite(osp.join(file_path, f'{dataset}_{numTrain}_polygon_result.png'), result)

def draw_polygon(all_class_polygon,save_path,dataset,numTrain):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as Patches
    import os.path as osp
    import os
    from skimage import io

    colors = color_map(dataset).tolist()
    image = io.imread(f'/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/{dataset}_rgb.png')
    #image = np.full_like(image,0)
    plt.axis('off')
    plt.imshow(image)

    for key,values in all_class_polygon.items():
        for polygon in values:
            polygon = np.array(polygon)
            color = np.array(colors[int(key)-1])/255
            plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec=color, linewidth=0.5))
            plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.', linewidth=1, markersize=2.5)   # WHU(linewidth=1, markersize=2.5)   linewidth=0.75, markersize=2

    file_path = osp.join(save_path, dataset)
    if not osp.exists(file_path):
        os.makedirs(file_path)
    impath = osp.join(file_path, f'{dataset}_{numTrain}_polygon_result.pdf')   # png

    plt.savefig(impath, bbox_inches='tight', pad_inches=0.0, dpi=800)
    plt.clf()