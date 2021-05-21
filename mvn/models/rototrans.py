from torch import nn

from mvn.models.resnet import MLPResNet
from mvn.models.layers import R6DBlock, RodriguesBlock, CaminoBlock


class RototransNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_joints = config.model.backbone.num_joints
        batch_norm = config.cam2cam.batch_norm
        drop_out = config.cam2cam.drop_out

        n_features = config.cam2cam.model.n_features

        f0 = 512
        f1 = 256
        f2 = 128
        f3 = 64

        self.backbone = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                2 * n_joints * 2, config.cam2cam.model.backbone.inner_size, 2, f0,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
        ])

        self.R_backbone = nn.Sequential(*[
            MLPResNet(
                f0, f0, config.cam2cam.model.roto.n_layers,
                f1,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            MLPResNet(
                f1, f1, config.cam2cam.model.roto.n_layers,
                f2,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            MLPResNet(
                f2, f2, config.cam2cam.model.roto.n_layers,
                f3,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            MLPResNet(
                f3, f3, config.cam2cam.model.roto.n_layers,
                6 if config.cam2cam.model.roto.parametrization == '6d' else 3,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            R6DBlock() if config.cam2cam.model.roto.parametrization == '6d' else RodriguesBlock()
        ])

        self.t_backbone = nn.Sequential(*[
            MLPResNet(
                f0, f0, config.cam2cam.model.trans.n_layers,
                f1,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            MLPResNet(
                f1, f1, config.cam2cam.model.trans.n_layers,
                f2,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            MLPResNet(
                f2, f2, config.cam2cam.model.trans.n_layers,
                f3,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            MLPResNet(
                f3, f3, config.cam2cam.model.trans.n_layers,
                3,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
        ])

    def forward(self, x):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        features = self.backbone(x)
        rot2rot = self.R_backbone(features)  # ~ (batch_size, 6)
        trans2trans = self.t_backbone(features)  # ~ (batch_size, 3)

        return rot2rot, trans2trans
