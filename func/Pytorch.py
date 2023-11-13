# -*- coding: utf-8 -*-
"""
@author: YuZhu
"""
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import cv2
from math import log


def sys_random_fixed(net_seed=2):
    torch.manual_seed(net_seed)  # cpu
    torch.cuda.manual_seed(net_seed)  # gpu
    torch.cuda.manual_seed_all(net_seed)  # gpu
    np.random.seed(net_seed)  # numpy
    random.seed(net_seed)  # numpy
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(net_seed)


def load_pretrained_model(model, path=''):
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


class MyDataset(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label


def MyDataLoader(data, label, batch_size=256, shuffle=False, drop_last=False, num_workers=4):
    mydataset = MyDataset(data, label)
    return DataLoader(mydataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


class MyDataset_idx(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index]
        idx = np.where(self.Label[index].flatten() > 0)[0]
        target = (self.Label[index].flatten())[idx] - 1
        return data, target.astype('long'), idx


def MyDataLoader_idx(data, label, batch_size=256, shuffle=False, drop_last=False, num_workers=4):
    mydataset = MyDataset_idx(data, label)
    return DataLoader(mydataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers)


class HRNet_(nn.Module):
    def __init__(self, conv_in, conv_out, conv_dim=64):
        super(HRNet_, self).__init__()
        self.conv_in, self.conv_dim, self.conv_out = conv_in, conv_dim, conv_out
        self.conv_init = nn.Sequential(
            nn.Conv2d(conv_in, conv_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim),
            # nn.GroupNorm(8, conv_dim),
            nn.ReLU(inplace=True),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim),
            # nn.GroupNorm(8, conv_dim),
            nn.ReLU(inplace=True),
        )
        self.downsample2x = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1),
        )
        self.downsample4x = nn.Sequential(
            nn.AvgPool2d(kernel_size=(5, 5), stride=4, padding=2),
        )
        self.conv_over = nn.Sequential(
            nn.Conv2d(conv_dim, conv_out, kernel_size=1, stride=1, padding=0),
        )

    def upsample(self, input, size, mode='bilinear'):
        return self.conv_3x3(nn.functional.interpolate(input, size=list(size)[-2:], mode=mode))

    def forward(self, x):
        conv0 = self.conv_init(x)

        conv1 = self.conv_3x3(conv0)

        conv2_1 = self.conv_3x3(conv1)
        conv2_2 = self.downsample2x(conv1)

        conv2_1 = self.conv_3x3(conv2_1)
        conv2_2 = self.conv_3x3(conv2_2)

        conv3_1 = conv2_1 + self.upsample(conv2_2, conv2_1.size())
        conv3_2 = self.downsample2x(conv2_1) + conv2_2

        conv3_1 = self.conv_3x3(conv3_1)  # 1*64*610*340
        conv3_2 = self.conv_3x3(conv3_2)  # 1*64*305*170

        conv4_1 = conv3_1 + self.upsample(conv3_2, conv3_1.size())
        conv4_2 = self.downsample2x(conv3_1) + conv3_2
        conv4_3 = self.downsample4x(conv3_1) + self.downsample2x(conv3_2)

        conv4_1 = self.conv_3x3(conv4_1)
        conv4_2 = self.conv_3x3(conv4_2)
        conv4_3 = self.conv_3x3(conv4_3)  # 1*64*153*85 向上取整

        conv5_1 = conv4_1 + self.upsample(conv4_2, conv4_1.size()) + self.upsample(conv4_3, conv4_1.size())
        conv5_2 = conv4_2 + self.downsample2x(conv4_1) + self.upsample(conv4_3, conv4_2.size())
        conv5_3 = self.downsample4x(conv4_1) + self.downsample2x(conv4_2) + conv4_3

        conv5_1 = self.conv_3x3(conv5_1)
        conv5_2 = self.conv_3x3(conv5_2)
        conv5_3 = self.conv_3x3(conv5_3)

        conv6 = conv5_2 + self.upsample(conv5_3, conv5_2.size())
        conv6 = self.conv_3x3(conv6)

        conv7 = conv5_1 + self.upsample(conv6, conv5_1.size())
        conv7 = self.conv_3x3(conv7)

        #conv_end = self.conv_over(conv7)  # 最后一层为线性层

        return conv7

class HRNet__(nn.Module):
    def __init__(self, conv_in, conv_out, conv_dim=64):
        super(HRNet__, self).__init__()
        self.conv_in, self.conv_dim, self.conv_out = conv_in, conv_dim, conv_out
        self.conv_init = nn.Sequential(
            nn.Conv2d(conv_in, conv_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim),
            # nn.ReLU(inplace=True),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim),
            # nn.ReLU(inplace=True),
        )
        self.conv_over = nn.Sequential(
            nn.Conv2d(conv_dim, conv_out, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        conv0 = self.conv_init(x)

        conv1 = self.conv_3x3(conv0)

        conv2 = self.conv_3x3(conv1)

        conv3 = self.conv_3x3(conv2)  # 1*64*610*340

        conv4 = self.conv_3x3(conv3)  # 1*64*610*340

        conv5 = self.conv_3x3(conv4)  # 1*64*610*340

        conv6 = self.conv_3x3(conv5)  # 1*64*610*340

        conv7 = self.conv_3x3(conv6)  # 1*64*610*340

        conv_end = self.conv_over(conv7)  # 最后一层为线性层

        return conv_end



class HRNet___(nn.Module):
    def __init__(self, conv_in, conv_out, conv_dim=64):
        super(HRNet___, self).__init__()
        self.conv_in, self.conv_dim, self.conv_out = conv_in, conv_dim, conv_out
        self.conv_init = nn.Sequential(
            nn.Conv2d(conv_in, conv_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_dim),
            # nn.ReLU(inplace=True),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(conv_in, conv_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_out),
            # nn.ReLU(inplace=True),
        )
        self.conv_over = nn.Sequential(
            nn.Conv2d(conv_in, conv_out, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        # conv_end = self.conv_init(x)
        #
        conv_end = self.conv_3x3(x)

        # conv_end = self.conv_over(x)  # 最后一层为线性层

        return conv_end


BN_MOMENTUM = 0.01


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM), nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRNet(nn.Module):
    def __init__(self, numLayers=305, numClasses=8):  # config):
        super(HRNet, self).__init__()
        # stem net
        self.conv1 = nn.Conv2d(numLayers, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)

        self.stage1_cfg = {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 1,
            'NUM_BLOCKS': [2],
            'NUM_CHANNELS': [32],
            'BLOCK': 'BOTTLENECK',
            'FUSE_METHOD': 'SUM',
        }

        self.stage2_cfg = {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 2,
            'NUM_BLOCKS': [2, 2],
            'NUM_CHANNELS': [32, 64],
            'BLOCK': 'BASIC',
            'FUSE_METHOD': 'SUM',

        }

        self.stage3_cfg = {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 3,
            'NUM_BLOCKS': [2, 2, 2],
            'NUM_CHANNELS': [32, 64, 96],
            'BLOCK': 'BASIC',
            'FUSE_METHOD': 'SUM',

        }

        # self.stage4_cfg = {
        #     'NUM_MODULES': 1,
        #     'NUM_BRANCHES': 4,
        #     'NUM_BLOCKS': [2, 2, 2, 2],
        #     'NUM_CHANNELS': [32, 64, 128, 256],
        #     'BLOCK': 'BASIC',
        #     'FUSE_METHOD': 'SUM',
        # }

        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        # num_channels = self.stage4_cfg['NUM_CHANNELS']
        # block = blocks_dict[self.stage4_cfg['BLOCK']]
        # num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        # self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        # self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=numClasses,
                      kernel_size=1, stride=1, padding=0)
        )

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels,
                                                num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # x_list = []
        # for i in range(self.stage4_cfg['NUM_BRANCHES']):
        #     if self.transition3[i] is not None:
        #         if i < self.stage3_cfg['NUM_BRANCHES']:
        #             x_list.append(self.transition3[i](y_list[i]))
        #         else:
        #             x_list.append(self.transition3[i](y_list[-1]))
        #     else:
        #         x_list.append(y_list[i])
        # x = self.stage4(x_list)
        x = y_list  # 后添加
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        # x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2], 1)
        # x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x

# def get_seg_model(cfg, **kwargs):
#     model = HRNet(cfg, **kwargs)
#     model = load_pretrained_model(model, cfg.MODEL.PRETRAINED)
#     return model

class SSAM(nn.Module):
    def __init__(self, channel, gamma=2, b=1, interact=True):
        super(SSAM, self).__init__()
        C = channel

        t = int(abs((log(C, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_channel = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.conv_spatial = nn.Sequential(

            nn.Conv2d(C, C // gamma, kernel_size=1, padding=0),
            nn.BatchNorm2d(C // gamma),
            nn.ReLU(inplace=True),

            nn.Conv2d(C // gamma, C//gamma, kernel_size=3, padding=1),
            nn.BatchNorm2d(C//gamma),
            nn.ReLU(inplace=True),

            nn.Conv2d(C//gamma, C//gamma, kernel_size=3, padding=1),
            nn.BatchNorm2d(C//gamma),
            nn.ReLU(inplace=True),

            nn.Conv2d(C//gamma, 1, kernel_size=1, padding=0)
        )
        self.sigmoid = nn.Sigmoid()

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.interact = interact

    def forward(self, x1, x2):
        if self.interact:
            y = x1 + x2
        else:
            y = x2

        y_channel = self.avg_pool(y)
        y_channel = self.conv_channel(y_channel.squeeze(-1).transpose(-1, -2).contiguous())
        y_channel = y_channel.transpose(-1 ,-2).contiguous().unsqueeze(-1)
        y_channel = self.sigmoid(y_channel)

        y_spatial = self.conv_spatial(y)
        y_spatial = self.sigmoid(y_spatial)

        Y = y_spatial + y_channel
        out = self.out_conv(x2 * (Y.expand_as(x2)))
        return out



class HSIVecNet(nn.Module):
    def __init__(self, conv_in, conv_out, conv_dim=64):
        super(HSIVecNet,self).__init__()
        self.backbone = HRNet_(conv_in, conv_out, conv_dim=64)
        self.mask_head = self._make_conv(conv_dim, conv_dim, conv_dim)
        self.jloc_head = self._make_conv(conv_dim, conv_dim, conv_dim)
        self.mask_att = SSAM(conv_dim,interact=False)
        self.jloc_att = SSAM(conv_dim,interact=False)

        self.mask_predictor = self._make_predictor(conv_dim, conv_out)
        self.jloc_predictor = self._make_predictor(conv_dim, 2)
        self.joff_head = self.joff_predictor(conv_dim, 2)
        self.m2j_att = SSAM(conv_dim,interact=True)
        self.j2m_att = SSAM(conv_dim,interact=True)

    def forward(self, x):
        feature_map = self.backbone(x)

        # baseline
        mask_feature__ = self.mask_head(feature_map)
        jloc_feature__ = self.jloc_head(feature_map)

        # mask_feature_ = mask_feature__ + jloc_feature__
        # jloc_feature_ = mask_feature__ + jloc_feature__
        # cross-task spectral-spacial attention module
        jloc_feature = self.m2j_att(mask_feature__,jloc_feature__)
        mask_feature = self.j2m_att(jloc_feature__,mask_feature__)

        mask_pred = self.mask_predictor(mask_feature)
        jloc_pred = self.jloc_predictor(jloc_feature)
        joff_pred = self.joff_head(feature_map)

        return mask_pred, jloc_pred, joff_pred

    def _make_conv(self, dim_in, dim_hid, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        return layer

    def _make_predictor(self, dim_in, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
        )
        return layer

    def joff_predictor(self, dim_in, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, int(dim_in / 4), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(dim_in / 4), dim_out, kernel_size=1, stride=1, padding=0)
        )
        return layer


def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)

    if mask is not None:
        t = (mask == 1).float()
        w = t.mean(0)
        mask = mask.float()
        mask[mask == 1] /= w
        mask[mask == 0] = 1
        mask = mask.reshape((-1, 1))
        #w[w == 0] = 1
        loss = loss * mask

    return loss.mean()


def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

def get_junctions(jloc, joff, topk = 300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc = jloc.reshape(-1)   #flatten->(1,128*128)
    joff = joff.reshape(2, -1)   #flatten->(2,128*128)

    scores, index = torch.topk(jloc, k=topk)   #get top x scores and its index
    y = (index // width).float() + torch.gather(joff[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()   #(2xN -> Nx2)

    return junctions[scores > th], scores[scores > th]

def get_pred_junctions(jloc_map, joff_map):
    #pred_nms = jloc_map
    pred_nms = non_maximum_suppression(jloc_map)  # probability map NMS
    topK = min(600, int((pred_nms > 0.5).float().sum().item()))  # select top points less than 300
    print('topK:',topK)
    juncs_pred, _ = get_junctions(pred_nms, joff_map, topk=topK)

    return juncs_pred.detach().cpu().numpy()  # (N,2)

def jloc_viz(jloc_map,dataset,Row, Col):
    pred_nms = non_maximum_suppression(jloc_map)  # probability map NMS
    jloc_pred = pred_nms > 0.5  # select top points less than 300
    imgDraw(jloc_pred.data.cpu().numpy().reshape(Row,Col),path='./viz/%s/%s' %('jloc',dataset),
            imgName='jloc_%s' % dataset)

def imgDraw(label, imgName, path='./pictures', show=True, dataset='indian_pines'):
    """
    功能：根据标签绘制RGB图
    输入：（标签数据，图片名）
    输出：RGB图
    备注：输入是2维数据，label有效范围[1,num]
    """
    from .Draw import color_map
    import matplotlib.pyplot as plt
    row, col = label.shape
    numClass = int(label.max())
    Y_RGB = np.zeros((row, col, 3)).astype('uint8')  # 生成相同shape的零数组
    Y_RGB[np.where(label == 0)] = [0, 0, 0]  # 对背景设置为黑色
    colors = color_map(dataset)
    for i in range(1, numClass + 1):  # 对有标签的位置上色
        try:
            Y_RGB[np.where(label == i)] = [255,255,255]
        except:
            Y_RGB[np.where(label == i)] = np.random.randint(0, 256, size=3)
    plt.axis("off")  # 不显示坐标
    if show:
        plt.imshow(Y_RGB)
    os.makedirs(path, exist_ok=True)
    plt.imsave(path + '/' + str(imgName) + '.png', Y_RGB)  # 分类结果图
    return Y_RGB


def get_poly(prop, mask_pred, junctions, mask_pred_argmax, prop_mean_area):
    from scipy.spatial.distance import cdist
    prop_mask = np.zeros_like(mask_pred).astype(np.uint8)
    prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1   #get a 0-1 mask of prop
    masked_instance = np.ma.masked_array(mask_pred, mask=(prop_mask != 1))   #use the mask to get every one instance
    score = masked_instance.mean()   #only get the mean of selected points in (prop_mask=1)
    im_h, im_w = mask_pred.shape
    contours, hierarchy = cv2.findContours(prop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   #get the counters of the instance  cv2.CHAIN_APPROX_NONE
    poly = []
    edge_index = []
    jid = 0

    for contour, h in zip(contours, hierarchy[0]):
        c = []
        if len(contour) < 3:
            continue
        if h[3] == -1:   #no inline outlines
            c = ext_c_to_poly_coco(contour, im_h, im_w)
        if h[3] != -1:   #id of inline outlines
            if cv2.contourArea(contour) >= 200:
                c = inn_c_to_poly_coco(contour, im_h, im_w)
            #c = inn_c_to_poly_coco(contour, im_h, im_w)
        if len(c) > 3:
            init_poly = c.copy()
            if len(junctions) > 0:
                cj_match_ = np.argmin(cdist(c, junctions), axis=1)   #match a junction for every counter point
                cj_dis = cdist(c, junctions)[np.arange(len(cj_match_)), cj_match_]   #get the distance of counter point and junctions
                threshold = np.median(cj_dis) / (1 + prop_mean_area / prop.area)
                #print(threshold)
                u, ind = np.unique(cj_match_[cj_dis < 1.5], return_index=True)   #get the earliest append point?
                if len(u) > 2:
                    ppoly = junctions[u[np.argsort(ind)]]   #get sorted junction of a counter
                    ppoly = np.concatenate((ppoly, ppoly[0].reshape(-1, 2)))
                    init_poly = ppoly

            init_poly = simple_polygon(init_poly, thres=10)
            #init_poly = Spectral_Consistency_Check(init_poly, mask_pred_argmax)

            poly.extend(init_poly.tolist())
            edge_index.append([i for i in range(jid, jid+len(init_poly)-1)])
            jid += len(init_poly)
    return np.array(poly), score, edge_index

def ext_c_to_poly_coco(ext_c, im_h, im_w):
    mask = np.zeros([im_h+1, im_w+1], dtype=np.uint8)
    polygon = np.int0(ext_c)
    cv2.drawContours(mask, [polygon.reshape(-1, 1, 2)], -1, color=1, thickness=-1)
    trans_prop_mask = mask.copy()
    f_y, f_x = np.where(mask == 1)
    trans_prop_mask[f_y + 1, f_x] = 1
    trans_prop_mask[f_y, f_x + 1] = 1
    trans_prop_mask[f_y + 1, f_x + 1] = 1
    contours, _ = cv2.findContours(trans_prop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)
    poly = np.concatenate((contour, contour[0].reshape(-1, 2)))
    new_poly = diagonal_to_square(poly)
    return new_poly

def inn_c_to_poly_coco(inn_c, im_h, im_w):
    mask = np.zeros([im_h + 1, im_w + 1], dtype=np.uint8)
    polygon = np.int0(inn_c)
    cv2.drawContours(mask, [polygon.reshape(-1, 1, 2)], -1, color=1, thickness=-1)   #padding mode
    trans_prop_mask = mask.copy()
    f_y, f_x = np.where(mask == 1)   #return pixel coord
    trans_prop_mask[f_y[np.where(f_y == min(f_y))], f_x[np.where(f_y == min(f_y))]] = 0
    trans_prop_mask[f_y[np.where(f_x == min(f_x))], f_x[np.where(f_x == min(f_x))]] = 0
    #trans_prop_mask[max(f_y), max(f_x)] = 1
    contours, _ = cv2.findContours(trans_prop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)[::-1]
    poly = np.concatenate((contour, contour[0].reshape(-1, 2)))
    #return poly
    new_poly = diagonal_to_square(poly)
    return new_poly

def simple_polygon(poly, thres=10):
    if (poly[0] == poly[-1]).all():
        poly = poly[:-1]
    lines = np.concatenate((poly, np.roll(poly, -1, axis=0)), axis=1)
    vec0 = lines[:, 2:] - lines[:, :2]
    vec1 = np.roll(vec0, -1, axis=0)
    vec0_ang = np.arctan2(vec0[:,1], vec0[:,0]) * 180 / np.pi
    vec1_ang = np.arctan2(vec1[:,1], vec1[:,0]) * 180 / np.pi
    lines_ang = np.abs(vec0_ang - vec1_ang)   #get ang of corner

    flag1 = np.roll((lines_ang > thres), 1, axis=0)
    flag2 = np.roll((lines_ang < 360 - thres), 1, axis=0)
    simple_poly = poly[np.bitwise_and(flag1, flag2)]
    simple_poly = np.concatenate((simple_poly, simple_poly[0].reshape(-1,2)))
    return simple_poly

def get_centerpoint(lis):
    area = 0.0
    x,y = 0.0,0.0

    a = len(lis)
    for i in range(a):
        lat = lis[i][0] #weidu
        lng = lis[i][1] #jingdu

        if i == 0:
            lat1 = lis[-1][0]
            lng1 = lis[-1][1]

        else:
            lat1 = lis[i-1][0]
            lng1 = lis[i-1][1]

        fg = (lat*lng1 - lng*lat1)/2.0
        area += fg
        x += fg*(lat+lat1)/3.0
        y += fg*(lng+lng1)/3.0

    x = x/area
    y = y/area

    return x,y

def component_polygon_area(poly):
    """Compute the area of a component of a polygon.
    Args:
        x (ndarray): x coordinates of the component
        y (ndarray): y coordinates of the component

    Return:
        float: the area of the component
    """
    x = poly[:,0]
    y = poly[:,1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))  # np.roll 意即“滚动”，类似移位操作
        # 注意这里的np.dot表示一维向量相乘

def Spectral_Consistency_Check(poly, mask_pred_argmax):

    _, H, W = mask_pred_argmax.shape
    print([[0, int(point[0]+0.5), int(point[1]+0.5)] for point in poly])
    class_list = [mask_pred_argmax[0, int(point[0]+0.5), int(point[1]+0.5)] for point in poly]   # get all point class_num
    print(class_list)
    main_class = max(set(class_list),key=class_list.count)   # get frequently append class_num
    #min_class = min(set(class_list),key=class_list.count)   # get min append class_num
    centroid_x,centroid_y = get_centerpoint(poly)   # get center location of polygon
    center_class = mask_pred_argmax[0, int(centroid_x+0.5), int(centroid_y+0.5)]   # get center point class as main_class
    #print(center_class)
    #class_count = len(set(class_list))

    idx_list = []
    poly_area = component_polygon_area(poly)
    for idx,class_num in enumerate(class_list):
        if class_num == main_class or class_num == center_class:
            idx_list.append(idx)
        else:
            new_poly = np.delete(poly, idx, axis=0)
            if len(new_poly) > 2:
                new_poly_area = component_polygon_area(new_poly)
                if new_poly_area < 0.9 * poly_area:
                    idx_list.append(idx)
    #poly[np.where(class_list == min_class)] = poly[np.where(class_list == min_class)] - (threshold/1.414)
    return poly[idx_list]




def diagonal_to_square(poly):
    new_c = []
    for id, p in enumerate(poly[:-1]):
        if (p[0] + 1 == poly[id + 1][0] and p[1] == poly[id + 1][1]) \
                or (p[0] == poly[id + 1][0] and p[1] + 1 == poly[id + 1][1]) \
                or (p[0] - 1 == poly[id + 1][0] and p[1] == poly[id + 1][1]) \
                or (p[0] == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
        elif (p[0] + 1 == poly[id + 1][0] and p[1] + 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0] + 1, p[1]])
        elif (p[0] - 1 == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0] - 1, p[1]])
        elif (p[0] + 1 == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0], p[1] - 1])
        else:
            new_c.append(p)
            new_c.append([p[0], p[1] + 1])
    new_poly = np.asarray(new_c)
    new_poly = np.concatenate((new_poly, new_poly[0].reshape(-1, 2)))
    return new_poly

def generate_polygon(prop, mask, juncs, mask_pred_argmax,prop_mean_area):
    poly, score, juncs_pred_index = get_poly(prop, mask, juncs, mask_pred_argmax, prop_mean_area)   #polygon_list scores edge_index:[range(points-1)]
    return poly, score, juncs_pred_index

def save_viz(image, polys, save_path, filename):
    import matplotlib.pyplot as plt
    import matplotlib.patches as Patches
    import os.path as osp
    colormap = (
        (0.6509803921568628, 0.807843137254902, 0.8901960784313725),
        (0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
        (0.984313725490196, 0.6039215686274509, 0.6),
        (0.8901960784313725, 0.10196078431372549, 0.10980392156862745),
        (0.9921568627450981, 0.7490196078431373, 0.43529411764705883),
        (1.0, 0.4980392156862745, 0.0),
        (0.792156862745098, 0.6980392156862745, 0.8392156862745098),
        (0.41568627450980394, 0.23921568627450981, 0.6039215686274509),
        (1.0, 1.0, 0.6),
        (0.6941176470588235, 0.34901960784313724, 0.1568627450980392))
    num_color = len(colormap)
    plt.axis('off')
    plt.imshow(image)

    for i, polygon in enumerate(polys):
        color = colormap[i % num_color]
        plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec=color, linewidth=1.5))
        plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.')

    file_path = osp.join(save_path, 'viz')
    if not osp.exists(file_path):
        os.makedirs(file_path)
    impath = osp.join(file_path, filename)

    plt.savefig(impath, bbox_inches='tight', pad_inches=0.0)
    plt.clf()

def poly_to_bbox(poly):
    """
    input: poly----2D array with points
    """
    lt_x = np.min(poly[:,0])
    lt_y = np.min(poly[:,1])
    w = np.max(poly[:,0]) - lt_x
    h = np.max(poly[:,1]) - lt_y
    return [float(lt_x), float(lt_y), float(w), float(h)]

def generate_coco_ann(all_class_polys, all_class_scores, img_id):
    sample_ann = []
    for key,values in all_class_polys.items():
        for i, polygon in enumerate(values):
            vec_poly = np.array(polygon).ravel().tolist()
            poly_bbox = poly_to_bbox(np.array(polygon))
            ann_per_building = {
                    'image_id': img_id,
                    'category_id': int(key),
                    'segmentation': [vec_poly],
                    'bbox': poly_bbox,
                    'score': float(all_class_scores[key][i]),
                }
            sample_ann.append(ann_per_building)

    return sample_ann

def generate_coco_ann_DP(all_class_polys, img_id):
    sample_ann = []
    for key,values in all_class_polys.items():
        for i, polygon in enumerate(values):
            vec_poly = np.array(polygon).ravel().tolist()
            poly_bbox = poly_to_bbox(np.array(polygon))
            ann_per_building = {
                    'image_id': img_id,
                    'category_id': int(key),
                    'segmentation': [vec_poly],
                    'bbox': poly_bbox,
                    'score': float(1),
                }
            sample_ann.append(ann_per_building)

    return sample_ann

def generate_coco_mask(mask_prop,mask_argmax, img_id):
    from pycocotools import mask as coco_mask
    from skimage.measure import label, regionprops
    Numclass, Row, Col = mask_prop.shape
    sample_ann = []
    for class_num in range(1,Numclass+1):
        per_class_mask = np.zeros((1, Row, Col))
        per_class_mask[np.where(mask_argmax == class_num)] = 1
        props = regionprops(label(per_class_mask.squeeze()))
        for prop in props:
            if ((prop.bbox[2] - prop.bbox[0]) > 0) & ((prop.bbox[3] - prop.bbox[1]) > 0):
                prop_mask = np.zeros_like(mask_prop[class_num,:,:], dtype=np.uint8)
                prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1

                masked_instance = np.ma.masked_array(mask_prop[class_num,:,:], mask=(prop_mask != 1))
                score = masked_instance.mean()
                encoded_region = coco_mask.encode(np.asfortranarray(prop_mask))
                ann_per_building = {
                    'image_id': img_id,
                    'category_id': int(class_num),
                    'segmentation': {
                        "size": encoded_region["size"],
                        "counts": encoded_region["counts"].decode()
                    },
                    'score': float(score),
                }
                sample_ann.append(ann_per_building)

    return sample_ann

def get_ce_weight(gt_1d):
    class_num = gt_1d.max() + 1
    num_list = []
    for i in range(class_num):
        num = sum(np.where(gt_1d == i,1,0))
        num_list.append(num)

    return num_list
