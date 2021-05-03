import traceback
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


# from 1812.07035 (https://github.com/papagina/RotationContinuity)
def normalize_vector(v, eps=1e-8):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(v_mag, torch.cuda.FloatTensor([eps]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v  # `nn.functional.normalize(v)`


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
    return out


# NN features (6D parametrization) -> rotation matrix
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0: 3]
    y_raw = ortho6d[:, 3: 6]

    x = nn.functional.normalize(x_raw)
    z = cross_product(x, y_raw)
    z = nn.functional.normalize(z)

    y = cross_product(z, x)
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)

    matrix = torch.cat((x, y, z), 2)
    return matrix  # 3 x 3


def l2_loss():
    def _f(gt, pred):
        return torch.pow(
            gt - pred,  # absolute error ~ (batch_size=8, ...)
            2
        ).mean()  # scalar
    return _f


def compute_geodesic_distance():
    def _f(m1, m2):
        batch_size = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # ~ (batch_size, 3, 3)

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        
        # bound [-1, 1]
        cos = torch.min(
            cos,
            torch.ones(batch_size).cuda()
        )
        cos = torch.max(
            cos,
            torch.ones(batch_size).cuda() * -1
        )

        theta = torch.acos(cos)
        return theta.mean()  # ~ (batch_size,)
    return _f


class RotoTransNetMLP(nn.Module):
    BN_MOMENTUM = 0.1

    def __init__(self, inner_size=128, n_joints=17, n_params=6):
        super().__init__()

        self.bn = nn.BatchNorm2d(num_features=2, momentum=self.BN_MOMENTUM)

        self.rot_backbone = nn.Sequential(
            nn.Linear(2 * n_joints * 2, inner_size),
            nn.LeakyReLU(),

            nn.Linear(inner_size, inner_size),
            nn.LeakyReLU(),

            nn.Linear(inner_size, inner_size),
            nn.LeakyReLU(),

            nn.Linear(inner_size, n_params)
        )

        self.trans_backbone = nn.Sequential(
            nn.Linear(2 * n_joints * 2, inner_size),
            nn.LeakyReLU(),

            nn.Linear(inner_size, inner_size),
            nn.LeakyReLU(),
            
            nn.Linear(inner_size, 3)  # 3D space
        )  # MLP

    def forward(self, batch):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        batch_size, n_joints = batch.shape[0], batch.shape[2]

        batch = self.bn(batch)  # 1 round of BatchNorm is always good
        batch = batch.view(batch_size, 2 * n_joints * 2)

        features = self.rot_backbone(batch)  # ~ (batch_size, n_params=6)
        rot2rot = compute_rotation_matrix_from_ortho6d(features)  # ~ (batch_size, 3, 3)

        trans2trans = self.trans_backbone(batch)  # ~ (batch_size, 3)

        return rot2rot, trans2trans


class RotoTransNetConv(nn.Module):
    BN_MOMENTUM = 0.1

    def __init__(self, inner_size=128, n_joints=17, n_params=6):
        super().__init__()

        self.bn = nn.BatchNorm2d(num_features=2, momentum=self.BN_MOMENTUM)

        # ~ 1 layer of UNet
        self.unet_layer = nn.Sequential(
            nn.BatchNorm2d(num_features=n_joints, momentum=self.BN_MOMENTUM),
            nn.Conv2d(n_joints, n_joints, 3, stride=1, padding=0, bias=True, padding_mode='zeros'),  # joints as channels
            nn.LeakyReLU(),

            nn.BatchNorm2d(num_features=n_joints, momentum=self.BN_MOMENTUM),
            nn.Conv2d(n_joints, n_joints, 3, stride=1, padding=0, bias=True, padding_mode='zeros'),  # joints as channels
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )  # ~ ResNet

        self.rot_backbone = nn.Sequential(
            nn.Linear(2 * n_joints * 14 * 14, inner_size),
            nn.LeakyReLU(),

            nn.Linear(inner_size, inner_size),
            nn.LeakyReLU(),

            nn.Linear(inner_size, n_params)
        )

        self.trans_backbone = nn.Sequential(
            nn.Linear(2 * n_joints * 14 * 14, inner_size),
            nn.LeakyReLU(),

            nn.Linear(inner_size, 3)
        )


    def forward(self, batch):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, width, height) """

        batch_size, n_joints = batch.shape[0], batch.shape[2]
        heatmaps_w, heatmaps_h = batch.shape[-2], batch.shape[-1]  # 32, 32

        batch = batch.view(batch_size * 2, n_joints, heatmaps_w, heatmaps_h)  # process each view with same weights
        features = self.unet_layer(batch)  # ~ batch_size * 2, n_joints, 14, 14
        features = features.view(
            batch_size, 2 * n_joints * features.shape[-2] * features.shape[-1]
        )

        rot2rot = self.rot_backbone(features)
        rot2rot = compute_rotation_matrix_from_ortho6d(rot2rot)  # ~ (batch_size, 3, 3)

        trans2trans = self.trans_backbone(features)  # ~ (batch_size, 3)

        return rot2rot, trans2trans
