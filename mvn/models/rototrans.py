import traceback
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


# from 1812.07035 (https://github.com/papagina/RotationContinuity)
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(
        v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda())
    )
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


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
    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)
    return matrix  # 3 x 3


def l2_loss(gt, pred):
    return torch.pow(
        gt - pred,  # absolute error ~ (batch_size=8, 3, 3)
        2
    ).mean()  # scalar


def compute_geodesic_distance(m1, m2):
    batch_size = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # ~ (batch_size=8, 3, 3)

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(
        cos,
        torch.autograd.Variable(torch.ones(batch_size).cuda())
    )
    cos = torch.max(
        cos,
        torch.autograd.Variable(torch.ones(batch_size).cuda()) * -1
    )

    theta = torch.acos(cos)
    return theta


class Roto6d(nn.Module):  # acts as SOTA
    def __init__(self, using_conv=False, n_params=6):
        super().__init__()

        if using_conv:
            self.backbone = ...  # todo, then expecting as input some heatmaps
        else:
            inner_size = 128
            self.backbone = nn.Sequential(
                nn.Linear(3 * 3, inner_size),
                nn.Linear(inner_size, n_params)
            )  # MLP

    def forward(self, batch):
        """ batch ~ many poses, i.e ~ (batch_size, 2, n_joints, 2D) """

        batch_size = batch.shape[0]
        features = self.backbone(
            None  # todo
        )
        return compute_rotation_matrix_from_ortho6d(features)
