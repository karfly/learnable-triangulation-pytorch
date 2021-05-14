from torch import nn

from mvn.models.mlp import MLP
from mvn.models.unet import MLUNet
from mvn.models.resnet import MLPResNet
from mvn.models.canonpose import CanonPose
from mvn.models.layers import R6DBlock


class RototransNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_joints = config.model.backbone.num_joints
        model = config.cam2cam.model.name
        inner_size = config.cam2cam.model.inner_size
        n_inner_layers = config.cam2cam.model.n_inner_layers
        batch_norm = config.cam2cam.batch_norm
        drop_out = config.cam2cam.drop_out

        in_features = 2 * n_joints * 2  # coming from a pair of 2D KPs

        if model == 'kiss':  # https://people.apache.org/~fhanik/kiss.html
            sizes = [in_features] + (n_inner_layers + 1) * [inner_size]

            # best so far: 5Lx512U, no BN
            self.R_backbone = nn.Sequential(*[
                nn.Flatten(),  # will be fed into a MLP
                MLP(
                    sizes + [6],  # need 6D parametrization of rotation matrix
                    batch_norm=batch_norm,
                    drop_out=drop_out,
                    activation=nn.LeakyReLU,
                ),
                R6DBlock()  # no params, just 6D -> R
            ])

            self.t_backbone = nn.Sequential(*[
                nn.Flatten(),  # will be fed into a MLP
                MLP(
                    sizes + [3],  # 3D world
                    batch_norm=batch_norm,
                    drop_out=drop_out,
                    activation=nn.LeakyReLU,
                ),
                nn.Linear(3, 3, bias=True),
            ])
        elif model == 'res':
            sizes = [in_features] + (n_inner_layers + 1) * [inner_size]

            self.R_backbone = nn.Sequential(*[
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

            self.t_backbone = nn.Sequential(*[
                nn.Flatten(),  # will be fed into a MLP
                MLPResNet(
                    in_features, inner_size, n_inner_layers,
                    3,
                    batch_norm=batch_norm,
                    drop_out=drop_out,
                    activation=nn.LeakyReLU,
                ),
            ])
        elif model == 'unet':
            self.R_backbone = nn.Sequential(*[
                nn.Flatten(),  # will be fed into a MLP
                MLUNet(
                    in_features,
                    6,
                    batch_norm=batch_norm,
                    drop_out=drop_out,
                    activation=nn.LeakyReLU
                ),
                R6DBlock()  # no params, just 6D -> R
            ])

            self.t_backbone = nn.Sequential(*[
                nn.Flatten(),  # will be fed into a MLP
                MLUNet(
                    in_features,
                    3,
                    batch_norm=batch_norm,
                    drop_out=drop_out,
                    activation=nn.LeakyReLU
                ),
            ])
        elif model == 'canonpose':
            self.R_backbone = nn.Sequential(*[
                nn.Flatten(),  # will be fed into a MLP
                CanonPose(
                    in_features,
                    6,
                    inner_size=inner_size,
                    batch_norm=batch_norm,
                    dropout=drop_out,
                    activation=nn.LeakyReLU
                ),
                R6DBlock()  # no params, just 6D -> R
            ])

            self.t_backbone = nn.Sequential(*[
                nn.Flatten(),  # will be fed into a MLP
                CanonPose(
                    in_features,
                    3,
                    inner_size=inner_size,
                    batch_norm=batch_norm,
                    dropout=drop_out,
                    activation=nn.LeakyReLU
                ),
                nn.Linear(3, 3, bias=True)
            ])
        else:
            raise ValueError('YOU HAVE TO SPECIFY A cam2cam MODEL!')

    def forward(self, batch):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        rot2rot = self.R_backbone(batch)  # ~ (batch_size, 6)
        trans2trans = self.t_backbone(batch)  # ~ (batch_size, 3)

        return rot2rot, trans2trans
