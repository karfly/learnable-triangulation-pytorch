from torch import nn
import torch

from mvn.models.resnet import MLPResNet
from mvn.models.layers import R6DBlock, RodriguesBlock


class RototransNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_views_comparing = 4
        self.n_pairs = self.n_views_comparing - 1

        n_joints = config.model.backbone.num_joints
        batch_norm = config.cam2cam.batch_norm
        drop_out = config.cam2cam.drop_out
        n_features = config.cam2cam.model.n_features

        self.backbone = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                in_features=self.n_views_comparing * n_joints * 2,
                inner_size=config.cam2cam.model.backbone.inner_size,
                n_inner_layers=config.cam2cam.model.backbone.n_layers,
                out_features=n_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
        ])

        # todo try specific norm
        n_params_per_R = 6 if config.cam2cam.model.roto.parametrization == '6d' else 3
        self.R_backbone = nn.Sequential(*[
            MLPResNet(
                in_features=n_features,
                inner_size=n_features,
                n_inner_layers=config.cam2cam.model.roto.n_layers,
                out_features=n_params_per_R * self.n_pairs,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
        ])
        self.r_model = R6DBlock() if config.cam2cam.model.roto.parametrization == '6d' else RodriguesBlock()

        # todo try specific norm
        self.t_backbone = nn.Sequential(*[
            MLPResNet(
                in_features=n_features,
                inner_size=n_features,
                n_inner_layers=config.cam2cam.model.trans.n_layers,
                out_features=3 * self.n_pairs,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
        ])

    def forward(self, x):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        batch_size = x.shape[0]
        features = self.backbone(x)  # batch_size, ...

        r_features = self.R_backbone(features)
        features_per_pair = r_features.shape[-1] // self.n_pairs
        r_features = r_features.view(
            batch_size, self.n_pairs, features_per_pair
        )
        rot2rot = torch.cat([
            self.r_model(r_features[batch_i]).unsqueeze(0)
            for batch_i in range(batch_size)
        ])  # batch_size, 3, (3 x 3)

        trans2trans = self.t_backbone(features)  # ~ (batch_size, 3)
        trans2trans = trans2trans.view(batch_size, self.n_pairs, 3)

        return rot2rot, trans2trans
