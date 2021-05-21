import torch
from torch import nn

import math

from torch.nn.modules.flatten import Flatten


class R6DBlock(nn.Module):
    """ https://arxiv.org/abs/1812.07035 """

    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize_vector(v, eps=1e-8):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))
        v_mag = torch.max(v_mag, torch.cuda.FloatTensor([eps]))
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        return v  # `nn.functional.normalize(v)`

    @staticmethod
    def cross_product(u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        return out

    def forward(self, x):
        x_raw = x[:, 0: 3]
        y_raw = x[:, 3: 6]

        x = nn.functional.normalize(x_raw)  # self.normalize_vector(x_raw)
        z = self.cross_product(x, y_raw)
        z = nn.functional.normalize(z)  # self.normalize_vector(z)

        y = self.cross_product(z, x)
        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)

        return torch.cat((x, y, z), 2)  # 3 x 3


class RodriguesBlock(nn.Module):
    """ https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/core/conversions.html#angle_axis_to_rotation_matrix """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    @staticmethod
    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def forward(self, angle_axis):
        """ https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation """

        # stolen from ceres/rotation.h

        _angle_axis = torch.unsqueeze(angle_axis, dim=1)
        theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
        theta2 = torch.squeeze(theta2, dim=1)

        # compute rotation matrices
        rotation_matrix_normal = self._compute_rotation_matrix(angle_axis, theta2)
        rotation_matrix_taylor = self._compute_rotation_matrix_taylor(angle_axis)

        # create mask to handle both cases
        eps = 1e-6
        mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
        mask_pos = (mask).type_as(theta2)
        mask_neg = (mask == False).type_as(theta2)  # noqa

        # create output pose matrix
        batch_size = angle_axis.shape[0]
        rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
        rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
        # fill output matrix with masked values
        rotation_matrix[..., :3, :3] = \
            mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
        
        return rotation_matrix[:, :3, :3]  # remove 0, 0, 0, 1 and 0s on the right => Nx3x3


# modified version of https://arxiv.org/abs/1709.01507, suitable for MLP
class SEBlock(nn.Module):
    def __init__(self, in_features, inner_size):
        super().__init__()

        self.excite = nn.Sequential(*[
            nn.Linear(in_features, inner_size, bias=True),
            nn.ReLU(inplace=False),

            nn.Linear(inner_size, in_features, bias=True),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        # it's already squeezed ...
        activation_map = self.excite(x)  # excite
        return torch.mul(
            activation_map,
            x
        )  # attend


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class CaminoBlock(nn.Module):
    def __init__(self, out_features):
        super().__init__()

        lst_units = [
            32*32,  # after conv will be 28 x 28 ...
            28*28,
            24*24,
            20*20,
            16*16,
        ]
        self.blocks = nn.Sequential(*[
            self._make_block(n_units)
            for n_units in lst_units
        ])

        self.head = nn.Sequential(*[
            nn.Flatten(),  # coming from a convolution
            nn.Linear(12*12, out_features, bias=True)
        ])

    @staticmethod
    def _make_block(n_units, activation=nn.LeakyReLU):
        return nn.Sequential(*[
            nn.Linear(n_units, n_units, bias=True),  # todo use MLPResNet
            activation(inplace=False),

            nn.Linear(n_units, n_units - 4, bias=True),
            activation(inplace=False),
        ])

    def forward(self, x):
        x = self.blocks(x)  # todo maybe with skip connections
        return self.head(x)
