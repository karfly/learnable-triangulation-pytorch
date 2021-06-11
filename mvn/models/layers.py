import torch
from torch import nn

from mvn.utils.tred import matrix_to_euler_angles
from mvn.models.resnet import MLPResNet


class R6DBlock(nn.Module):
    """ https://arxiv.org/abs/1812.07035 """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_raw = x[:, 0: 3]
        y_raw = x[:, 3: 6]

        x = nn.functional.normalize(x_raw)  # b1
        z = nn.functional.normalize(torch.cross(x, y_raw))  # b2
        y = torch.cross(z, x)  # b1 x b2

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)

        return torch.cat([x, y, z], 2)  # 3 x 3


class R2DBlock(nn.Module):
    """ https://arxiv.org/abs/1812.07035 """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_raw = x[:, 0: 2]
        y_raw = x[:, 2: 4]

        x = nn.functional.normalize(x_raw)  # b1

        dot = torch.cat([
            torch.dot(x[i], y_raw[i]).unsqueeze(0)
            for i in range(x.shape[0])
        ])
        y = nn.functional.normalize(
            y_raw -\
                torch.cat([
                    (
                        x[i] * dot[i]
                    ).unsqueeze(0)
                    for i in range(x.shape[0])
                ])
        )  # b2

        x = x.view(-1, 2, 1)
        y = y.view(-1, 2, 1)

        return torch.cat([x, y], 2)  # 2 x 2


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


class DepthBlock(nn.Module):
    def __init__(self, in_features, inner_size, n_inner_layers, n2predict, batch_norm, drop_out, activation):
        super().__init__()

        self.bb = MLPResNet(
            in_features=in_features,
            inner_size=inner_size,
            n_inner_layers=n_inner_layers,
            out_features=1 * n2predict,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            final_activation=None,
            init_weights=False
        )

    def forward(self, x):
        return self.bb(x)


class TranslationFromAnglesBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.param = R2DBlock()

    def forward(self, batched_features):
        dev = batched_features.device

        batched_Rx = self.param(batched_features[:, :4])
        batched_Rz = self.param(batched_features[:, 4: 8])
        batched_d = batched_features[:, -1].view(-1, 1)

        batched_Rz_in_3D = torch.eye(3)\
            .repeat(batched_features.shape[0], 1)\
            .view(batched_features.shape[0], 3, 3).to(dev)
        batched_Rz_in_3D[:, :2, :2] = batched_Rz

        batched_Rx_in_3D = torch.eye(3)\
            .repeat(batched_features.shape[0], 1)\
            .view(batched_features.shape[0], 3, 3).to(dev)
        batched_Rx_in_3D[:, 1:, 1:] = batched_Rx

        batched_R_in_3D = torch.bmm(
            batched_Rz_in_3D,  # first turn around Z ...
            batched_Rx_in_3D  # ... then around X
        )

        batched_d_in_3D = torch.zeros(3)\
            .repeat(batched_features.shape[0], 1)\
            .view(batched_features.shape[0], 3, 1).to(dev)
        batched_d_in_3D[:, 2] = batched_d  # as Z

        return torch.bmm(
            batched_R_in_3D, batched_d_in_3D
        ).view(-1, 1, 3)


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
