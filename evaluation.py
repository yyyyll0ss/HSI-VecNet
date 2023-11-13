import torch
import argparse
from pycocotools.coco import COCO
from multiprocess import Pool

""" 设置系统参数及CPU """
parser = argparse.ArgumentParser(description='相关参数 可在命令行编辑')
parser.add_argument('--dataset', type=str, default='WHU_Hi_HongHu', choices=['indian_pines','salinas_valley', 'WHU_Hi_HongHu'],help='数据集')
parser.add_argument('--random_seed', type=int, default=0, help='固定随机数种子')
#parser.add_argument('--checkpoint', type=str, default='./model_WHU_Hi_HongHu_0.2_4.pth', help='model checkpoint')
args = parser.parse_args()
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print('Your device is a', device)

from func.Metric import ContourEval, PolisEval, compute_IoU_cIoU


def polis_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    polisEval = PolisEval(gt_coco, dt_coco)
    polis_avg = polisEval.evaluate()
    return polis_avg

def max_angle_error_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    contour_eval = ContourEval(gt_coco, dt_coco)
    pool = Pool(processes=20)
    max_angle_diffs = contour_eval.evaluate(pool=pool)
    print('Mean max tangent angle error(MTA): ', max_angle_diffs.mean())
    return max_angle_diffs.mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/datasets/PaviaU/PaviaU_coco.json")   #'indian_pines','salinas_valley','WHU_Hi_HongHu','WHU_Hi_LongKou','PaviaU','Houston'
    parser.add_argument("--dt-file", default="/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/output/PaviaU/PaviaU_0.2.json")
    parser.add_argument("--eval-type", default='ciou', choices=["polis", "angle", "ciou"])
    args = parser.parse_args()
    is_DP = 'DP' in args.dt_file

    eval_type = args.eval_type
    gt_file = args.gt_file
    dt_file = args.dt_file

    polis_avg = polis_eval(gt_file, dt_file)
    angle_error = max_angle_error_eval(gt_file, dt_file)
    miou, mciou = compute_IoU_cIoU(dt_file, gt_file, is_DP)
    print(f'mIoU:{miou:.4f}, mC-IoU:{mciou:.4f}, polis:{polis_avg:.4f}, max_angle_error:{angle_error:.4f}')



