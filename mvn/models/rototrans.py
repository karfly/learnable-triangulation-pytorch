from torch import nn

from mvn.models.resnet import MLPResNet
from mvn.models.layers import R6DBlock


class RototransNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_joints = config.model.backbone.num_joints
        batch_norm = config.cam2cam.batch_norm
        drop_out = config.cam2cam.drop_out

        n_features = config.cam2cam.model.n_features
        activation = lambda: nn.LeakyReLU(negative_slope=1e-2, inplace=False)

        self.R_backbone = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                2 * n_joints * 2, n_features,
                config.cam2cam.model.roto.n_layers,
                6,  # 6D parametrization of matrix
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
                init_weights=True
            ),
            R6DBlock()
        ])

        self.t_backbone = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                2 * n_joints * 2, n_features,
                config.cam2cam.model.trans.n_layers,
                3,  # 3D camspace t vector
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
                init_weights=True
            ),
        ])

    def forward(self, x):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        rot2rot = self.R_backbone(x)  # ~ (batch_size, 6)
        trans2trans = self.t_backbone(x)  # ~ (batch_size, 3)

        return rot2rot, trans2trans
