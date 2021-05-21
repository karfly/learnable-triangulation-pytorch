# Reference: https://github.com/microsoft/human-pose-estimation.pytorch

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
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

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
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

        out += residual
        out = self.relu(out)

        return out


class GlobalAveragePoolingHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=False),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=False),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)

        batch_size, n_channels = x.shape[:2]
        x = x.view((batch_size, n_channels, -1))
        x = x.mean(dim=-1)

        out = self.head(x)

        return out


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


class PoseResNet(nn.Module):
    def __init__(self, block, layers, num_joints,
                 num_input_channels=3,
                 deconv_with_bias=False,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 final_conv_kernel=1,
                 alg_confidences=False,
                 vol_confidences=False
                 ):
        super().__init__()

        self.num_joints = num_joints
        self.num_input_channels = num_input_channels
        self.inplanes = 64

        self.deconv_with_bias = deconv_with_bias
        self.num_deconv_layers, self.num_deconv_filters, self.num_deconv_kernels = num_deconv_layers, num_deconv_filters, num_deconv_kernels
        self.final_conv_kernel = final_conv_kernel

        self.conv1 = nn.Conv2d(
            num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if alg_confidences:
            self.alg_confidences = GlobalAveragePoolingHead(512 * block.expansion, num_joints)

        if vol_confidences:
            self.vol_confidences = GlobalAveragePoolingHead(512 * block.expansion, 32)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            self.num_deconv_layers,
            self.num_deconv_filters,
            self.num_deconv_kernels,
        )

        self.final_layer = nn.Conv2d(
            in_channels=self.num_deconv_filters[-1],
            out_channels=self.num_joints,
            kernel_size=self.final_conv_kernel,
            stride=1,
            padding=1 if self.final_conv_kernel == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is != len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is != len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=False))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        alg_confidences = None
        if hasattr(self, "alg_confidences"):
            alg_confidences = self.alg_confidences(x)

        vol_confidences = None
        if hasattr(self, "vol_confidences"):
            vol_confidences = self.vol_confidences(x)

        x = self.deconv_layers(x)
        features = x

        x = self.final_layer(x)
        heatmaps = x

        return heatmaps, features, alg_confidences, vol_confidences


def get_pose_net(config, num_deconv_filters=(256, 256, 256), device='cuda:0'):
    block_class, layers = resnet_spec[config.model.backbone.num_layers]
    if config.model.backbone.style == 'caffe':
        block_class = Bottleneck_CAFFE

    model = PoseResNet(
        block_class, layers, config.model.backbone.num_joints,
        num_input_channels=3,
        deconv_with_bias=False,
        num_deconv_layers=3,
        num_deconv_filters=num_deconv_filters,
        num_deconv_kernels=(4, 4, 4),
        final_conv_kernel=1,
        alg_confidences=config.model.backbone.alg_confidences,
        vol_confidences=config.model.backbone.vol_confidences
    )

    need_backbone = True
    if config.model.cam2cam_estimation and config.cam2cam.using_gt:
        need_backbone = False

    if need_backbone and config.model.backbone.init_weights:
        print("Loading pretrained weights from: {}".format(config.model.backbone.checkpoint))
        model_state_dict = model.state_dict()

        pretrained_state_dict = torch.load(config.model.backbone.checkpoint, map_location=device)

        if 'state_dict' in pretrained_state_dict:
            pretrained_state_dict = pretrained_state_dict['state_dict']

        prefix = "module."

        new_pretrained_state_dict = {}
        for k, v in pretrained_state_dict.items():
            if k.replace(prefix, "") in model_state_dict and v.shape == model_state_dict[k.replace(prefix, "")].shape:
                new_pretrained_state_dict[k.replace(prefix, "")] = v
            elif k.replace(prefix, "") == "final_layer.weight":  # TODO
                print("  reiniting final layer filters:", k)

                o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:, :, :, :])
                nn.init.xavier_uniform_(o)
                n_filters = min(o.shape[0], v.shape[0])
                o[:n_filters, :, :, :] = v[:n_filters, :, :, :]

                new_pretrained_state_dict[k.replace(prefix, "")] = o
            elif k.replace(prefix, "") == "final_layer.bias":
                print("  reiniting final layer biases:", k)
                o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:])
                nn.init.zeros_(o)
                n_filters = min(o.shape[0], v.shape[0])
                o[:n_filters] = v[:n_filters]

                new_pretrained_state_dict[k.replace(prefix, "")] = o

        not_inited_params = set(map(lambda x: x.replace(prefix, ""), pretrained_state_dict.keys())) - set(new_pretrained_state_dict.keys())
        if len(not_inited_params) > 0:
            print("Parameters [{}] were not inited".format(not_inited_params))

        model.load_state_dict(new_pretrained_state_dict, strict=False)
        print("Successfully loaded pretrained weights for backbone")

    return model
