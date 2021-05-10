import traceback
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

from mvn.models.vgg import make_virgin_vgg
from mvn.models.layers import ExtractEverythingLayer, MLP, View
from mvn.models.resnet import MLPResNetBlock


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

    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)

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
    def __init__(self, config):
        super().__init__()

        n_joints = config.model.backbone.num_joints
        inner_size = config.cam2cam.mlp.inner_size
        n_inner_layers = config.cam2cam.mlp.n_inner_layers
        kiss = True  # https://people.apache.org/~fhanik/kiss.html

        if kiss:
            sizes = [2 * n_joints * 2] + (n_inner_layers + 1) * [inner_size]

            self.roto_extractor = nn.Sequential(*[
                View((-1, sizes[0])),
                nn.LayerNorm((sizes[0])),
                MLP(
                    sizes + [6],  # need 6D parametrization of rotation matrix
                    batch_norm=config.cam2cam.batch_norm,
                    drop_out=config.cam2cam.drop_out,
                    activation=nn.ReLU,
                    init_weights=False
                )
            ])

            self.trans_extractor = nn.Sequential(*[
                View((-1, sizes[0])),
                nn.LayerNorm((sizes[0])),
                MLP(
                    sizes + [3],  # 3D world
                    batch_norm=config.cam2cam.batch_norm,
                    drop_out=config.cam2cam.drop_out,
                    activation=nn.LeakyReLU,
                    init_weights=False
                )
            ])
        else:
            self.roto_extractor = nn.Sequential(*[
                View((-1, 2 * n_joints * 2)),  # because it will be fed into a MLP
                MLPResNetBlock(
                    n_inner_layers,
                    2 * n_joints * 2,
                    inner_size,
                    6,  # need 6D parametrization of rotation matrix
                    batch_norm=config.cam2cam.batch_norm,
                    mlp=lambda a, b: nn.Linear(a, b, bias=False),
                )
            ])

            self.trans_extractor = nn.Sequential(*[
                View((-1, 2 * n_joints * 2)),  # because it will be fed into a MLP
                MLPResNetBlock(
                    n_inner_layers,
                    2 * n_joints * 2,
                    inner_size,
                    3,  # 3D world
                    batch_norm=config.cam2cam.batch_norm,
                    mlp=lambda a, b: nn.Linear(a, b, bias=False)
                )
            ])

    def forward(self, batch):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        features = self.roto_extractor(batch)  # ~ (batch_size, 6)
        rot2rot = compute_rotation_matrix_from_ortho6d(features)  # ~ (batch_size, 3, 3)

        trans2trans = self.trans_extractor(batch)  # ~ (batch_size, 3)

        return rot2rot, trans2trans


class RotoTransNetConv(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_joints = config.model.backbone.num_joints
        inner_size = config.cam2cam.mlp.inner_size
        n_inner_layers = config.cam2cam.mlp.n_inner_layers
        sizes = (n_inner_layers + 1) * [inner_size]

        self.roto_extractor = nn.Sequential(*[
            make_virgin_vgg(
                config.cam2cam.vgg,
                batch_norm=config.cam2cam.batch_norm,
                in_channels=2 * n_joints,
                kernel_size=3,
                num_classes=sizes[0],
                init_weights=False
            ),  # ~ encoder
            MLP(
                sizes + [6],  # need 6D parametrization of rotation matrix
                batch_norm=config.cam2cam.batch_norm,
                drop_out=config.cam2cam.drop_out,
                activation=nn.LeakyReLU,
                init_weights=False
            )  # ~ decoder
        ])

        self.trans_extractor = nn.Sequential(*[
            make_virgin_vgg(
                config.cam2cam.vgg,
                batch_norm=config.cam2cam.batch_norm,
                in_channels=2 * n_joints,
                kernel_size=3,
                num_classes=sizes[0],
                init_weights=False
            ),  # ~ encoder
            MLP(
                sizes + [3],  # 3D world
                batch_norm=config.cam2cam.batch_norm,
                drop_out=config.cam2cam.drop_out,
                activation=nn.LeakyReLU,
                init_weights=False
            )  # ~ decoder
        ])

    def forward(self, batch):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, width, height) """

        # stack each view "vertically"
        batch = torch.cat([
            batch[:, 0, ...],
            batch[:, 1, ...],
        ], dim=1)  # ~ batch_size, 17 * 2, 32, 32

        features = self.roto_extractor(batch)
        rot2rot = compute_rotation_matrix_from_ortho6d(features)  # ~ (batch_size, 3, 3)

        trans2trans = self.trans_extractor(batch)

        return rot2rot, trans2trans
