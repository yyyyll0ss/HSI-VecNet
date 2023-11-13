from func.Pytorch import HSIVecNet, get_pred_junctions, generate_polygon, save_viz, \
    generate_coco_ann, generate_coco_mask, jloc_viz # 使用HRNet
import torch
from func.Hyper import matDataLoad, calcAccuracy, split_dataset, pointsLoad, draw_poly_result, draw_polygon
import numpy as np
import argparse
from skimage.measure import label, regionprops
import skimage.io as io
import os.path as osp
import os
import cv2
from func.Draw import imgDraw
import json

""" 设置系统参数及CPU """
parser = argparse.ArgumentParser(description='相关参数 可在命令行编辑')
parser.add_argument('--dataset', type=str, default='indian_pines', choices=['indian_pines','salinas_valley', 'WHU_Hi_LongKou','WHU_Hi_HongHu','PaviaU','Houston'],help='数据集')
parser.add_argument('--random_seed', type=int, default=0, help='固定随机数种子')
parser.add_argument('--numTrain', type=int, default=0.2, help='每类训练样本数量')
parser.add_argument('--checkpoint', type=str, default=f'./checkpoint/PaviaU/model_indian_pines_0.2.pth', help='model checkpoint')
args = parser.parse_args()
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print('Your device is a', device)

""" 获取原始数据及基本信息 """
data_hsi = matDataLoad(f'datasets/{args.dataset}/' + args.dataset + '.mat')
gt_hsi = matDataLoad(f'datasets/{args.dataset}/' + args.dataset + '_gt.mat')
gt_points_hsi = pointsLoad(f'datasets/{args.dataset}/' + args.dataset + '_coco.json')   #jloc joff
gt_jloc = gt_points_hsi['jloc']   # ground truth of junction location (1,H,W)
gt_joff = gt_points_hsi['joff']   # ground truth of junction offset (2,H,W)

gt_hsi_1D = gt_hsi.flatten()
gt_jloc_1D = gt_jloc.flatten()
gt_joff_1D = gt_joff.flatten(1).transpose(1,0)   #->(2,(H*W)) -> ((H*w),2)
#train_idx, test_idx = split_dataset(gt_hsi_1D, gt_jloc_1D, args.numTrain)  # 分割训练集和测试集,每类数量一致,positive and negetive samples equal

Row, Col, Layers = data_hsi.shape
NumClass = gt_hsi.max()
Model_in = Layers  # 模型输入
Model_out = NumClass + 1

""" 数据转换 """
# create input tensor (B C H W)
data_hsi_tensor = data_hsi.transpose(2, 0, 1)  # 转为band,row,col C H W
data_hsi_tensor = data_hsi_tensor[np.newaxis, :, :, :]  # 1*103*610*340
data_hsi_tensor = torch.from_numpy(data_hsi_tensor).to(device)  # 转化为tensor数据

"""create model and get output"""
#create model
model = HSIVecNet(Model_in, Model_out, conv_dim=64).to(device)
model.load_state_dict(torch.load(args.checkpoint))
mask_pred, jloc_pred, joff_pred = model(data_hsi_tensor)   # 1*class*H*W 1*2*H*W 1*2*H*W
mask_pred_argmax = mask_pred[0].softmax(0).argmax(0).reshape((1, Row, Col)).detach().cpu().numpy()   # ->class*H*W ->1*H*W
mask_pred = mask_pred[0].softmax(0).detach().cpu().numpy()   # (class+1)*H*W convert to numpy
jloc_pred_prob = jloc_pred[0].softmax(0)[1, :, :].reshape((1, Row, Col))  # 2*H*W -> 1*H*W
joff_pred = joff_pred[0].sigmoid() - 0.5
jloc_pred_argmax = jloc_pred[0].softmax(0).argmax(0).reshape((1, Row, Col)).detach().cpu().numpy()

juncs_pred = get_pred_junctions(jloc_pred_prob, joff_pred)
jloc_viz(jloc_pred_prob,args.dataset,Row, Col)

all_juncs = juncs_pred
all_masks = mask_pred_argmax
all_class_polygons = {}
all_class_scores = {}

for class_num in range(1, NumClass+1):
    per_class_mask = np.zeros((1,Row, Col))
    per_class_mask[np.where(mask_pred_argmax == class_num)] = 1
    Polys, Scores = [], []
    props = regionprops(label(per_class_mask.squeeze()))
    prop_mean_area = np.median(np.array([prop.area for prop in props if prop.area > 0]))

    for prop in props:
        if prop.area > 20:   # indian/salinas(40),HongHu/LongKou(100)
            poly, scores, juncs = generate_polygon(prop, mask_pred[class_num,:,:], juncs_pred, mask_pred_argmax, prop_mean_area)
            if len(poly) != 0:
                Polys.append(poly.tolist())
                Scores.append(scores)

    all_class_polygons[str(class_num)] = Polys
    all_class_scores[str(class_num)] = Scores

"""save polygon viz result"""
save_path = './end2end_polygon_result'
#draw_poly_result(all_class_polygons,save_path,args.dataset,args.numTrain)
draw_polygon(all_class_polygons,save_path,args.dataset,args.numTrain)
np.save('./pictures/%s_%s_result.npy' % (args.dataset, args.numTrain),mask_pred_argmax.reshape(Row, Col))
imgDraw(mask_pred_argmax.reshape(Row, Col), imgName="./_%s_%s_result" % (args.dataset, args.numTrain),dataset=args.dataset)

"""generate coco style result to json"""
image_result = generate_coco_ann(all_class_polygons, all_class_scores, 0)
image_masks = generate_coco_mask(mask_pred, mask_pred_argmax, 0)

# creat polygon save path
poly_path_ = f'./output/{args.dataset}'
if not osp.exists(poly_path_):
    os.makedirs(poly_path_)
poly_path = osp.join(poly_path_,f'{args.dataset}_{args.numTrain}.json')

# creat mask save path
mask_path_ = f'./output/{args.dataset}'
if not osp.exists(mask_path_):
    os.makedirs(mask_path_)
mask_path = osp.join(mask_path_,f'{args.dataset}_{args.numTrain}_mask.json')

# save two style result to json file
with open(poly_path, 'w') as _out:
    json.dump(image_result, _out)

with open(mask_path, 'w') as _out:
    json.dump(image_masks, _out)


data_hsi = matDataLoad(f'datasets/{args.dataset}/' + args.dataset + '.mat')
gt_hsi = matDataLoad(f'datasets/{args.dataset}/' + args.dataset + '_gt.mat')
gt_points_hsi = pointsLoad(f'datasets/{args.dataset}/' + args.dataset + '_coco.json')   #jloc joff
gt_jloc = gt_points_hsi['jloc']   # ground truth of junction location (1,H,W)
gt_joff = gt_points_hsi['joff']   # ground truth of junction offset (2,H,W)

gt_hsi_1D = gt_hsi.flatten()
gt_jloc_1D = gt_jloc.flatten()
gt_joff_1D = gt_joff.flatten(1).transpose(1,0)   #->(2,(H*W)) -> ((H*w),2)

mask_train_idx, mask_test_idx, mask_num_list = split_dataset(gt_hsi_1D, gt_jloc_1D, args.numTrain, is_mask=True)  # 分割训练集和测试集,每类数量一致,positive and negetive samples equal
jloc_train_idx, jloc_test_idx, jloc_num_list = split_dataset(gt_hsi_1D, gt_jloc_1D, args.numTrain, is_mask=False)  # 分割训练集和测试集,每类数量一致,positive and negetive samples equal

# creat labels for different task
mask_train_target = gt_hsi_1D[mask_train_idx] #- 1  # 真训练集标签
mask_test_target = gt_hsi_1D[mask_test_idx] #- 1  # 真验证集标签
jloc_train_target = gt_jloc_1D[jloc_train_idx]   # 0 or 1
jloc_test_target = gt_jloc_1D[jloc_test_idx]   # 0 or 1
joff_train_target = gt_joff_1D[jloc_train_idx]   # offset
joff_test_target = gt_joff_1D[jloc_test_idx]   # offset

OA_mask, Kappa_mask, AA_mask = calcAccuracy(mask_pred_argmax.flatten()[mask_test_idx], mask_test_target)
OA_junc, Kappa_junc, AA_junc = calcAccuracy(jloc_pred_argmax.flatten()[jloc_test_idx], jloc_test_target.data.cpu().numpy())

print("mask-OA:%.3f" % OA_mask)
print("mask-AA:%.3f" % AA_mask.mean())
print("mask-Kappa:%.3f" % Kappa_mask)
print('junc-OA:%.3f' % OA_junc)
print("junc-AA:%.3f" % AA_junc.mean())
print("junc-Kappa:%.3f" % Kappa_junc)



