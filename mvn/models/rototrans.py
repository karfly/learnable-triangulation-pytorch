from torch import nn
import torch

from mvn.models.resnet import MLPResNet
from mvn.models.layers import R6DBlock, RodriguesBlock


class RotoTransCombiner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rotations, translations):
        batch_size = rotations.shape[0]
        n_views = rotations.shape[1]

        if translations.shape[-1] == 1:  # predicted just distance
            trans = torch.cat([  # ext.t in each view
                torch.zeros(batch_size, n_views, 2, 1).to(rotations.device),
                translations.unsqueeze(-1),  # ~ batch_size, | comparisons |, 1, 1
            ], dim=-2)  # vstack => ~ batch_size, | comparisons |, 3, 1
        else:
            trans = translations.unsqueeze(-1)

        roto_trans = torch.cat([  # ext (not padded) in each view
            rotations, trans
        ], dim=-1)  # hstack => ~ batch_size, | comparisons |, 3, 4
        roto_trans = torch.cat([  # padd each view
            roto_trans,
            torch.cuda.DoubleTensor(
                [0, 0, 0, 1]
            ).repeat(batch_size, n_views, 1, 1)
        ], dim=-2)  # hstack => ~ batch_size, | comparisons |, 3, 4

        # master_cam_i = 0  # first view acting as master
        return torch.cat([
            torch.cat([
                # torch.mm(
                #     roto_trans[batch_i, target_view],
                #     torch.inverse(roto_trans[batch_i, master_cam_i])
                # ).unsqueeze(0)  # 1 x 4 x 4
                roto_trans[batch_i, view_i].unsqueeze(0)
                for view_i in range(n_views)
            ]).unsqueeze(0)
            for batch_i in range(batch_size)
        ])  # todo tensored


class RotoTransNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_views_comparing = 4
        self.n_pairs = self.n_views_comparing - 1

        n_joints = config.model.backbone.num_joints
        batch_norm = config.cam2cam.model.batch_norm
        drop_out = config.cam2cam.model.drop_out
        n_features = config.cam2cam.model.n_features

        if config.cam2cam.model.type == 'mlp':  # what else?
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
                # can be helpful sometimes nn.BatchNorm1d(n_features),
            ])

        n_params_per_R = 6 if config.cam2cam.model.R.parametrization == '6d' else 3
        if config.cam2cam.model.type == 'mlp':
            self.R_backbone = nn.Sequential(*[
                MLPResNet(
                    in_features=n_features,
                    inner_size=n_features,
                    n_inner_layers=config.cam2cam.model.R.n_layers,
                    out_features=n_params_per_R * self.n_views_comparing,
                    batch_norm=batch_norm,
                    drop_out=drop_out,
                    activation=nn.LeakyReLU,
                ),
            ])
        self.r_model = R6DBlock() if config.cam2cam.model.R.parametrization == '6d' else RodriguesBlock()

        self.td = 1 if config.cam2cam.data.pelvis_in_origin else 3  # just d
        if config.cam2cam.model.type == 'mlp':
            self.t_backbone = nn.Sequential(*[
                MLPResNet(
                    in_features=n_features,
                    inner_size=n_features,
                    n_inner_layers=config.cam2cam.model.t.n_layers,
                    out_features=self.td * self.n_views_comparing,
                    batch_norm=batch_norm,
                    drop_out=drop_out,
                    activation=nn.LeakyReLU,
                ),
            ])
        # todo self.t_model = 

        self.combiner = RotoTransCombiner()

    def forward(self, x):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        batch_size = x.shape[0]
        features = self.backbone(x)  # batch_size, ...

        R_feats = self.R_backbone(features)
        features_per_pair = R_feats.shape[-1] // self.n_views_comparing
        R_feats = R_feats.view(
            batch_size, self.n_views_comparing, features_per_pair
        )
        rots = torch.cat([  # ext.R in each view
            self.r_model(R_feats[batch_i]).unsqueeze(0)
            for batch_i in range(batch_size)
        ])  # ~ batch_size, | comparisons |, (3 x 3)

        t_feats = self.t_backbone(features)  # ~ (batch_size, 3)
        trans = t_feats.view(batch_size, self.n_views_comparing, self.td)  # ~ batch_size, | comparisons |, 1 = ext.d for each view

        return self.combiner(rots, trans)
