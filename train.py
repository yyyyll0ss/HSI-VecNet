# -*- coding: utf-8 -*-
import numpy as np
import torch
from func.Hyper import matDataLoad, calcAccuracy, felzenszwalb_hsi, split_dataset, dataNormalize, print_f,pointsLoad
from func.Draw import imgDraw, imgDraw_idx, imgDraw_mask  # 不绘图可注释
from func.Others import timeRecord
from func.Pytorch import HRNet_, HSIVecNet, sigmoid_l1_loss, get_ce_weight  # 使用HRNet
import cv2
import argparse
import time
import skimage.io as io

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'gray', 'indianred',
          'chocolate', 'tan', 'skyblue', 'olive', 'lime', 'teal', 'hotpink', 'purple', 'k']

""" Parameter settings """
parser = argparse.ArgumentParser(description='相关参数 可在命令行编辑')
parser.add_argument('--dataset', type=str, default='indian_pines', choices=['indian_pines','salinas_valley','WHU_Hi_HongHu','WHU_Hi_LongKou','PaviaU', 'Houston'],help='数据集')
parser.add_argument('--numTrain', type=int, default=0.2, help='每类训练样本数量')
parser.add_argument('--lr', type=float, default=0.1, help='学习率')
parser.add_argument('--lr_dec_epoch', type=int, default=100, help='每N轮衰减一次学习率')
parser.add_argument('--lr_dec_rate', type=float, default=0.9, help='学习率衰减率')
parser.add_argument('--numEpoch', type=int, default=600, help='网络训练轮数')
parser.add_argument('--minEpoch', type=int, default=100, help='网络至少训练轮数')
parser.add_argument('--random_seed', type=int, default=0, help='固定随机数种子')

args = parser.parse_args()
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print('Your device is a', device)

""" load the origin data and get the basic information"""
data_hsi = matDataLoad(f'datasets/{args.dataset}/' + args.dataset + '.mat')
gt_hsi = matDataLoad(f'datasets/{args.dataset}/' + args.dataset + '_gt.mat')
gt_points_hsi = pointsLoad(f'datasets/{args.dataset}/' + args.dataset + '_coco.json')   #jloc joff
gt_jloc = gt_points_hsi['jloc']   # ground truth of junction location (1,H,W)
gt_joff = gt_points_hsi['joff']   # ground truth of junction offset (2,H,W)

gt_hsi_1D = gt_hsi.flatten()
gt_jloc_1D = gt_jloc.flatten()
gt_joff_1D = gt_joff.flatten(1).transpose(1,0)   # ->(2,(H*W)) -> ((H*w),2)

""" split train and test datasets"""
mask_train_idx, mask_test_idx, mask_num_list = split_dataset(gt_hsi_1D, gt_jloc_1D, args.numTrain, is_mask=True)  # split train and test dataset,positive and negetive samples equal
jloc_train_idx, jloc_test_idx, jloc_num_list = split_dataset(gt_hsi_1D, gt_jloc_1D, args.numTrain, is_mask=False)  # split train and test dataset,positive and negetive samples equal
# print('mask_train_num:',len(mask_train_idx),'mask_test_num:',len(mask_test_idx),'Train/Test:',len(mask_train_idx)/(len(mask_train_idx)+len(mask_test_idx)))
# print('jloc_train_num:',len(jloc_train_idx),'jloc_train_num:',len(jloc_test_idx),'Train/Test:',len(jloc_train_idx)/(len(jloc_train_idx)+len(jloc_test_idx)))

Row, Col, Layers = data_hsi.shape
NumClass = gt_hsi.max()
Model_in = Layers  # 模型输入
Model_out = NumClass + 1
Overfit = (NumClass * args.numTrain - 1) / (NumClass * args.numTrain)

""" convert data into tensor type """
# create input tensor (B C H W)
data_hsi_tensor = data_hsi.transpose(2, 0, 1)  # 转为band,row,col C H W
data_hsi_tensor = data_hsi_tensor[np.newaxis, :, :, :]  # 1*103*610*340
data_hsi_tensor = torch.from_numpy(data_hsi_tensor).to(device)  # 转化为tensor数据

# creat labels for different task
mask_train_target = gt_hsi_1D[mask_train_idx] # - 1  # 真训练集标签
mask_test_target = gt_hsi_1D[mask_test_idx] # - 1  # 真验证集标签
jloc_train_target = gt_jloc_1D[jloc_train_idx]   # 0 or 1
jloc_test_target = gt_jloc_1D[jloc_test_idx]   # 0 or 1
joff_train_target = gt_joff_1D[jloc_train_idx]   # offset
joff_test_target = gt_joff_1D[jloc_test_idx]   # offset

# convert to tensor
mask_label_tensor = torch.from_numpy(mask_train_target).long().to(device)
jloc_label_tensor = jloc_train_target.to(device)
joff_label_tensor = joff_train_target.to(device)

timeRecord()  # 记录时间

""" creat model """
model = HSIVecNet(Model_in, Model_out, conv_dim=64).to(device)

criterion = torch.nn.CrossEntropyLoss()   # weight=mask_ce_weight
Balanced_ce_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1,100]).float().to('cuda:0'))   # weight=jloc_ce_weight
# optimizer = torch.optim.SGD(model.parameters(), lr=0.12, momentum=0.9)  # , weight_decay=0.001)   #Honghu 0.2 yuan 0.12
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)   # we use adam to train (lr=lr=5e-4, weight_decay=0) Honghu
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)  # 动态学习率
model.train()
flag = 0  # 结束训练标志位
BestAcc = 0.9
for batch_idx in range(args.numEpoch):
    ''' 计算训练集损失 '''
    optimizer.zero_grad()
    mask_pred, jloc_pred, joff_pred = model(data_hsi_tensor)  # 输入1*103*610*340.输出 1*9*610*340 1*2*H*W 1*2*H*W

    #calculate the mask_pred ce-loss
    mask_pred = mask_pred[0].permute(1, 2, 0)  # 转为610*340*9
    mask_pred = mask_pred.view(-1, Model_out)  # 展平为207400*9
    #mask_pred = torch.nn.functional.softmax(mask_pred, dim=-1)  # 归一化 (H*W)*class
    mask_loss = criterion(mask_pred[mask_train_idx], mask_label_tensor)  # 计算真实训练集的损失

    #calculate the jloc_pred ce-loss
    jloc_pred = jloc_pred[0].permute(1, 2, 0)  # 转为H*W*2
    jloc_pred = jloc_pred.view(-1, 2)  # 展平为(H*W)*2
    #jloc_pred = torch.nn.functional.softmax(jloc_pred, dim=-1)  # 归一化  (H*W)*2
    jloc_loss = Balanced_ce_loss(jloc_pred[jloc_train_idx], jloc_label_tensor)  # 计算真实训练集的损失

    # calculate the joff_pred l1-loss
    joff_pred = joff_pred[0].permute(1, 2, 0)  # 转为H*W*2
    joff_pred = joff_pred.view(-1, 2)  # 展平为(H*W)*2
    joff_loss = sigmoid_l1_loss(joff_pred[jloc_train_idx], joff_label_tensor, -0.5, jloc_label_tensor)

    all_loss = 4 * mask_loss + 8 * jloc_loss + 0.25 * joff_loss
    all_loss.backward()
    optimizer.step()
    scheduler.step()  # 实时更新学习率

    ''' print training information '''
    # mask accuricy
    mask_pred_ = torch.nn.functional.softmax(mask_pred, dim=-1)
    mask_im_target = torch.argmax(mask_pred_, -1).data.cpu().numpy()  # 获得最大值索引numpy形式，207400*1
    mask_acc_train = np.sum(mask_im_target[mask_train_idx] == mask_train_target) / len(mask_train_target)
    mask_acc_test = np.sum(mask_im_target[mask_test_idx] == mask_test_target) / len(mask_test_target)

    # jloc accuricy
    jloc_pred_ = torch.nn.functional.softmax(jloc_pred, dim=-1)
    jloc_im_target = torch.argmax(jloc_pred_, -1).data.cpu().numpy()
    jloc_acc_train = np.sum(jloc_im_target[jloc_train_idx] == jloc_train_target.cpu().numpy()) / len(jloc_train_target.cpu().numpy())
    jloc_acc_test = np.sum(jloc_im_target[jloc_test_idx] == jloc_test_target.cpu().numpy()) / len(jloc_test_target.cpu().numpy())

    print('Epoch:', batch_idx + 1, '/', args.numEpoch,
          'all_Loss:%.3f' % all_loss.item(),
          'mask_Loss:%.3f' % mask_loss.item(),
          'jloc_Loss:%.3f' % jloc_loss.item(),
          'joff_Loss:%.3f' % joff_loss.item(),
          'lr:%.3f' % (scheduler.get_lr()[0]),
          'Acc on train-mask:%.3f' % mask_acc_train,
          'Acc on test-mask:%.5f' % mask_acc_test,
          'Acc on train-jloc:%.3f' % jloc_acc_train,
          'Acc on test-jloc:%.5f' % jloc_acc_test,
          )
    imgDraw(mask_im_target.reshape(Row, Col), path='./image/%s/%s' % ('mask', args.dataset),
            imgName='idx_%d' % batch_idx)  # 实时绘制当前结果图
    #jloc_im_target[np.where(gt_hsi_1D == 0)] = 0
    imgDraw(jloc_im_target.reshape(Row, Col), path='./image/%s/%s' % ('jloc', args.dataset),
            imgName='idx_%d' % batch_idx)  # 实时绘制当前结果图

    # if mask_acc_train == 1:
    #     flag += 1
    #     if flag > 20:
    #         if batch_idx >= args.minEpoch:  # 过拟合停止训练
    #             break

    if mask_acc_train > BestAcc:
        BestAcc = mask_acc_train
        torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/model_{args.dataset}_{args.numTrain}.pth')

    if (batch_idx + 1) % 50 == 0:
        message = "%s_%s_epoch:%s_time:%s" % (args.dataset, args.numTrain, batch_idx + 1,
                                                 time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        filename = "%s_%s" % (args.dataset, args.numTrain)
        print(message)
        OA_mask, Kappa_mask, AA_mask = calcAccuracy(mask_im_target[mask_test_idx], mask_test_target)
        OA_junc, Kappa_junc, AA_junc = calcAccuracy(jloc_im_target[jloc_test_idx], jloc_test_target.data.cpu().numpy())
        # for i in AA:
        #     print_f("%.3f" % i, filename)
        print_f("mask-OA:%.3f" % OA_mask, filename)
        print_f("mask-AA:%.3f" % AA_mask.mean(), filename)
        print_f("mask-Kappa:%.3f" % Kappa_mask, filename)
        print_f("Time:%.3f" % timeRecord(show=False), filename)
        print_f('junc-OA:%.3f' % OA_junc, filename)
        print_f("junc-AA:%.3f" % AA_junc.mean(), filename)
        print_f("junc-Kappa:%.3f" % Kappa_junc, filename)
        print_f(' ')
        print_f(' ')
        print_f(' ')


torch.save(model.state_dict(),f'./checkpoint/{args.dataset}/model_{args.dataset}_{args.numTrain}.pth')
""" save result with background or not """
#mask_im_target += 1
imgDraw(mask_im_target.reshape(Row, Col), imgName="./_%s_%s_result" % (args.dataset, args.numTrain),dataset=args.dataset)
#save pre_mask image
pre_reslut_label = mask_im_target.reshape(Row, Col)
np.save('./pictures/%s_%s_result.npy' % (args.dataset, args.numTrain),pre_reslut_label)
mask_im_target[np.where(gt_hsi_1D == 0)] = 0  # 背景像素不显示
imgDraw(mask_im_target.reshape(Row, Col), imgName="./%s_%s_result_no_back" % (args.dataset, args.numTrain),dataset=args.dataset)

model.eval()
timeRecord(type='part', show=False)
with torch.no_grad():
    optimizer.zero_grad()
    output = model(data_hsi_tensor)

timeRecord(type='part')

"""output result vectorization"""

