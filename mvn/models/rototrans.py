import traceback
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

from mvn.models.vgg import make_virgin_vgg
from mvn.models.layers import make_MLP


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


class ExtractEverythingLayer(nn.Module):
    def __init__(self, in_channels, n_out_features, mlp_sizes, batch_norm=False):
        super().__init__()

        self.encoder = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=True,
            padding_mode='zeros'
        )

        self.mlp_in = mlp_sizes[0]
        self.mlp = make_MLP(
            mlp_sizes,
            n_classes=n_out_features,
            batch_norm=batch_norm,
            activation=nn.LeakyReLU
        )

    def forward(self, x):
        x = self.encoder(x)

        x = x.view(-1, self.mlp_in)  # ready for MLP
        x = self.mlp(x)

        return x


class RotoTransNetMLP(nn.Module):
    BN_MOMENTUM = 0.1

    def __init__(self, config, inner_size=128):
        super().__init__()

        n_joints = config.model.backbone.num_joints
        sizes = [
            n_joints,
            config.cam2cam.inner_size,
            config.cam2cam.inner_size // 2,
            config.cam2cam.inner_size // 4,
        ]

        self.roto_extractor = ExtractEverythingLayer(
            n_joints,
            6,  # need 6D parametrization of rotation matrix
            sizes,
            batch_norm=config.cam2cam.batch_norm
        )

        self.trans_extractor = ExtractEverythingLayer(
            n_joints,
            3,  # need 3D vector
            sizes,
            batch_norm=config.cam2cam.batch_norm
        )

    def forward(self, batch):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        batch_size, n_joints = batch.shape[0], batch.shape[2]
        batch = batch.view(batch_size, n_joints, 2, 2)

        features = self.roto_extractor(batch)  # ~ (batch_size, 6)
        rot2rot = compute_rotation_matrix_from_ortho6d(features)  # ~ (batch_size, 3, 3)

        trans2trans = self.trans_extractor(batch)  # ~ (batch_size, 3)

        return rot2rot, trans2trans


class RotoTransNetConv(nn.Module):
    BN_MOMENTUM = 0.1

    def __init__(self, config):
        super().__init__()

        n_joints = config.model.backbone.num_joints

        # inspired by http://arxiv.org/abs/1905.10711
        sizes = [
            config.cam2cam.inner_size,
            config.cam2cam.inner_size // 2,
            config.cam2cam.inner_size // 4,
        ]

        # todo try smaller backbone
        self.roto_encoder = make_virgin_vgg(
            config.cam2cam.backbone,
            batch_norm=config.cam2cam.batch_norm,
            in_channels=n_joints,
            num_classes=sizes[0]
        )
        self.roto_decoder = make_MLP(
            sizes,
            n_classes=6,
            batch_norm=config.cam2cam.batch_norm,
            activation=nn.LeakyReLU
        )  # need 6D parametrization of rotation matrix

        # todo try smaller backbone
        self.trans_encoder = make_virgin_vgg(
            config.cam2cam.backbone,
            batch_norm=config.cam2cam.batch_norm,
            in_channels=n_joints,
            num_classes=sizes[0]
        )

        self.trans_decoder = make_MLP(
            sizes,
            n_classes=3,
            batch_norm=config.cam2cam.batch_norm,
            activation=nn.LeakyReLU
        )

    def forward(self, batch):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, width, height) """

        # stack each view vertically
        batch = torch.cat([
            batch[:, 0, ...],
            batch[:, 1, ...],
        ], dim=2)  # ~ 3, 17, 64, 32

        features = self.roto_encoder(batch)
        features = self.roto_decoder(features)
        rot2rot = compute_rotation_matrix_from_ortho6d(features)  # ~ (batch_size, 3, 3)

        features = self.trans_encoder(batch)
        features = self.trans_decoder(features)
        trans2trans = features

        return rot2rot, trans2trans
