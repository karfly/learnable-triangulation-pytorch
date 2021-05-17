from torch import nn
from torch.nn.modules import activation

from mvn.models.mlp import MLP
from mvn.models.unet import MLUNet
from mvn.models.resnet import MLPResNet
from mvn.models.canonpose import CanonPose
from mvn.models.layers import R6DBlock


# todo split
class RotoNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_joints = config.model.backbone.num_joints
        inner_size = config.cam2cam.model.inner_size
        n_inner_layers = config.cam2cam.model.n_inner_layers
        batch_norm = config.cam2cam.batch_norm
        drop_out = config.cam2cam.drop_out

        in_features = 2 * n_joints * 2  # coming from a pair of 2D KPs

        self.backbone = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                in_features, inner_size, n_inner_layers,
                6,  # need 6D parametrization of rotation matrix
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            R6DBlock()  # no params, just 6D -> R
        ])

    def forward(self, batch):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        return self.backbone(batch)


class TransNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_joints = config.model.backbone.num_joints
        inner_size = config.cam2cam.model.inner_size
        n_inner_layers = config.cam2cam.model.n_inner_layers
        batch_norm = config.cam2cam.batch_norm
        drop_out = config.cam2cam.drop_out

        in_features = 2 * n_joints * 2  # coming from a pair of 2D KPs

        self.backbone = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                in_features, inner_size, n_inner_layers,
                3,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
        ])

    def forward(self, batch):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        return self.backbone(batch)


class RototransNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_joints = config.model.backbone.num_joints
        batch_norm = config.cam2cam.batch_norm
        drop_out = config.cam2cam.drop_out

        n_features = config.cam2cam.model.n_features
        n_refine_features = config.cam2cam.model.n_refine_features
        activation = lambda: nn.LeakyReLU(negative_slope=1e-2, inplace=False)

        self.backbone = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                2 * n_joints * 2,  # coming from a pair of 2D KPs
                config.cam2cam.model.backbone.inner_size,
                config.cam2cam.model.backbone.n_layers,
                n_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
            ),
        ])  # shared

        self.R_backbone = nn.Sequential(*[
            MLPResNet(
                n_features, n_features,
                config.cam2cam.model.roto.n_layers,
                n_refine_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
            ),
            MLPResNet(
                n_refine_features, n_refine_features,
                config.cam2cam.model.roto.n_layers,
                6,  # 6D parametrization of matrix
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
            ),
            R6DBlock()
        ])

        self.t_backbone = nn.Sequential(*[
            MLPResNet(
                n_features, n_features,
                2,
                n_refine_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
            ),
            MLPResNet(
                n_refine_features, n_refine_features,
                2,
                3,  # 3D camspace t vector
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
            ),
        ])


    def forward(self, batch):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        features = self.backbone(batch)
        rot2rot = self.R_backbone(features)  # ~ (batch_size, 6)
        trans2trans = self.t_backbone(features)  # ~ (batch_size, 3)

        return rot2rot, trans2trans
