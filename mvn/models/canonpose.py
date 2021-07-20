import torch
import torch.nn as nn

from mvn.models.layers import RodriguesBlock


class res_block(nn.Module):
    def __init__(self, inner_size=1024):
        super().__init__()

        self.l1 = nn.Linear(inner_size, inner_size)
        self.l2 = nn.Linear(inner_size, inner_size)

    def forward(self, x):
        inp = x
        x = nn.LeakyReLU()(self.l1(x))
        x = nn.LeakyReLU()(self.l2(x))
        x += inp
        return x


class CanonPose(nn.Module):
    """ adapted from https://github.com/bastianwandt/CanonPose """

    def __init__(self, inner_size=1024, n_joints=17):
        super().__init__()

        self.upscale = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            nn.Linear((2 + 1) * n_joints, inner_size)  # joints + confidences
        ])
        self.res_common = res_block(inner_size=inner_size)
        self.res_pose1 = res_block(inner_size=inner_size)
        self.res_pose2 = res_block(inner_size=inner_size)
        self.res_cam1 = res_block(inner_size=inner_size)
        self.res_cam2 = res_block(inner_size=inner_size)
        self.pose3d = nn.Linear(inner_size, 3 * n_joints)
        self.enc_rot = nn.Linear(inner_size, 3)

        self.n_joints = n_joints
        self.rodrigues = RodriguesBlock()

    def _forward_pose(self, x):
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        return self.pose3d(xp).reshape(-1, self.n_joints, 3)

    def _forward_camera(self, x):
        xc = nn.LeakyReLU()(self.res_cam1(x))
        xc = nn.LeakyReLU()(self.res_cam2(xc))
        xc = self.enc_rot(xc)

        # angles are in axis angle notation -> use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
        return self.rodrigues(xc)

    def forward(self, p2d, conf):
        x = torch.cat((p2d, conf.unsqueeze(-1)), axis=-1)
        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))

        x_pose = self._forward_pose(x)  # pose path
        xc = self._forward_camera(x)  # camera path
        return x_pose, xc
