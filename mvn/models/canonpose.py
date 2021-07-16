import torch
import torch.nn as nn


class res_block(nn.Module):
    def __init__(self):
        super().__init__()

        inner_size = 1024
        self.l1 = nn.Linear(inner_size, inner_size)
        self.l2 = nn.Linear(inner_size, inner_size)

    def forward(self, x):
        inp = x
        x = nn.LeakyReLU()(self.l1(x))
        x = nn.LeakyReLU()(self.l2(x))
        x += inp
        return x


class Lifter(nn.Module):
    """ adapted from https://github.com/bastianwandt/CanonPose """

    def __init__(self):
        super().__init__()

        inner_size = 1024

        self.upscale = nn.Linear(32+16, inner_size)
        self.res_common = res_block()
        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        self.res_cam1 = res_block()
        self.res_cam2 = res_block()
        self.pose3d = nn.Linear(inner_size, 48)
        self.enc_rot = nn.Linear(inner_size, 3)

    def forward(self, p2d, conf):
        x = torch.cat((p2d, conf), axis=1)  # = flatten

        x = self.upscale(x)
        x = nn.LeakyReLU()(self.res_common(x))

        # pose path
        xp = nn.LeakyReLU()(self.res_pose1(x))
        xp = nn.LeakyReLU()(self.res_pose2(xp))
        x_pose = self.pose3d(xp)

        # camera path
        xc = nn.LeakyReLU()(self.res_cam1(x))
        xc = nn.LeakyReLU()(self.res_cam2(xc))
        xc = self.enc_rot(xc)

        return x_pose, xc
