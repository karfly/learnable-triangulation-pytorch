from torch import nn

from mvn.models.resnet import MLPResNet
from mvn.models.layers import R6DBlock


class RototransNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_joints = config.model.backbone.num_joints
        batch_norm = config.cam2cam.batch_norm
        drop_out = config.cam2cam.drop_out

        n_features = 512
        n_out_features = 256

        self.backbone = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                2 * n_joints * 2, config.cam2cam.model.backbone.inner_size, 2, n_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
        ])

        self.R_backbone = nn.Sequential(*[
            MLPResNet(
                n_features, n_features, config.cam2cam.model.roto.n_layers, n_out_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            MLPResNet(
                n_out_features, n_out_features, config.cam2cam.model.roto.n_layers, 6,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            R6DBlock()
        ])

        self.t_backbone = nn.Sequential(*[
            MLPResNet(
                n_features, n_features, 2, n_out_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            MLPResNet(
                n_out_features, n_out_features, 2, 3,
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
