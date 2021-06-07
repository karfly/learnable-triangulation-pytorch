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
                torch.abs(translations).unsqueeze(-1),  # ~ batch_size, | comparisons |, 1, 1
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

        return torch.cat([
            torch.cat([
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
        self.scale_t = config.cam2cam.postprocess.scale_t

        n_joints = config.model.backbone.num_joints
        batch_norm = config.cam2cam.model.batch_norm
        drop_out = config.cam2cam.model.drop_out
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
            nn.BatchNorm1d(n_features),
        ])

        n_params_per_R, self.R_param = None, None
        if config.cam2cam.model.R.parametrization == '6d':
            n_params_per_R = 6
            self.R_param = R6DBlock()
        elif config.cam2cam.model.R.parametrization == 'rodrigues':
            n_params_per_R = 3
            self.R_param = RodriguesBlock()
        
        self.R_model = MLPResNet(
            in_features=n_features,
            inner_size=n_features,
            n_inner_layers=config.cam2cam.model.R.n_layers,
            out_features=n_params_per_R * self.n_views_comparing,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=nn.LeakyReLU,
        )

        self.td = 1 if config.cam2cam.data.pelvis_in_origin else 3  # just d
        self.t_model = MLPResNet(
            in_features=n_features,
            inner_size=n_features,
            n_inner_layers=config.cam2cam.model.t.n_layers,
            out_features=self.td * self.n_views_comparing,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=nn.LeakyReLU,
        )

        self.combiner = RotoTransCombiner()

    def forward(self, x):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        batch_size = x.shape[0]
        features = self.backbone(x)  # batch_size, ...

        R_feats = self.R_model(features)
        features_per_pair = R_feats.shape[-1] // self.n_views_comparing
        R_feats = R_feats.view(
            batch_size, self.n_views_comparing, features_per_pair
        )
        rots = torch.cat([  # ext.R in each view
            self.R_param(R_feats[batch_i]).unsqueeze(0)
            for batch_i in range(batch_size)
        ])  # ~ batch_size, | n_predictions |, (3 x 3)

        t_feats = self.t_model(features)  # ~ (batch_size, 3)
        trans = t_feats.view(batch_size, self.n_views_comparing, self.td)  # ~ batch_size, | n_predictions |, 1 = ext.d for each view
        trans = trans * self.scale_t

        return self.combiner(rots, trans)


class Cam2camNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_views = 4
        self.n_master2other_pairs = self.n_views - 1  # 0 -> 1, 0 -> 2 ...
        self.scale_t = config.cam2cam.postprocess.scale_t

        n_joints = config.model.backbone.num_joints
        batch_norm = config.cam2cam.model.batch_norm
        drop_out = config.cam2cam.model.drop_out
        n_features = config.cam2cam.model.n_features

        self.backbone = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                in_features=self.n_views * n_joints * 2,
                inner_size=config.cam2cam.model.backbone.inner_size,
                n_inner_layers=config.cam2cam.model.backbone.n_layers,
                out_features=n_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
                final_activation=nn.LeakyReLU,
            ),
            nn.BatchNorm1d(n_features),
        ])

        if config.cam2cam.model.master.R.parametrization == '6d':
            n_params_per_R = 6
            self.master_R_param = R6DBlock()
        elif config.cam2cam.model.master.R.parametrization == 'rodrigues':
            n_params_per_R = 3
            self.master_R_param = RodriguesBlock()
        else:
            n_params_per_R, self.master_R_param = None, None

        self.master_R_model = nn.Sequential(*[
            MLPResNet(
                in_features=n_features,
                inner_size=n_features,
                n_inner_layers=config.cam2cam.model.master.R.n_layers,
                out_features=n_params_per_R,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),  # master.R predictor
        ])

        t_params = 1 if config.cam2cam.data.pelvis_in_origin else 3  # just d
        self.master_t_model = nn.Sequential(*[
            MLPResNet(
                in_features=n_features,
                inner_size=n_features,
                n_inner_layers=config.cam2cam.model.master.t.n_layers,
                out_features=t_params,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),  # master.t predictor
        ])

        self.cam2cam_backbone = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                in_features=n_features,
                inner_size=n_features,
                n_inner_layers=config.cam2cam.model.master2others.backbone.n_layers,
                out_features=n_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
                final_activation=nn.LeakyReLU,
            ),
            nn.BatchNorm1d(n_features),
        ])

        if config.cam2cam.model.master2others.R.parametrization == '6d':
            n_params_per_R = 6
            self.master2others_R_param = R6DBlock()
        elif config.cam2cam.model.master2others.R.parametrization == 'rod':
            n_params_per_R = 3
            self.master2others_R_param = RodriguesBlock()
        else:
            n_params_per_R, self.master2others_R_param = None, None

        out_features = n_params_per_R * self.n_master2other_pairs
        self.master2others_R_model = nn.Sequential(*[
            MLPResNet(
                in_features=n_features,
                inner_size=n_features,
                n_inner_layers=config.cam2cam.model.master2others.R.n_layers,
                out_features=out_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
        ])

        out_features = 3 * self.n_master2other_pairs
        self.master2others_t_model = nn.Sequential(*[
            MLPResNet(
                in_features=n_features,
                inner_size=n_features,
                n_inner_layers=config.cam2cam.model.master2others.t.n_layers,
                out_features=out_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
        ])

        self.combiner = RotoTransCombiner()  # what else ???

    def _forward_masters(self, features):
        batch_size = features.shape[0]
        R_feats = self.master_R_model(features)
        R_feats = R_feats.view(
            batch_size, -1
        )
        Rs = self.master_R_param(R_feats)  # ~ batch_size, (3 x 3)

        t_feats = self.master_t_model(features)
        ts = t_feats.view(batch_size, -1)
        ts = ts * self.scale_t

        return self.combiner(
            Rs.unsqueeze(1),
            ts.unsqueeze(1),
        ).view(batch_size, 4, 4)  # master's extrinsics

    def _forward_master2others(self, features):
        batch_size = features.shape[0]
        more_features = self.cam2cam_backbone(features)

        R_feats = self.master2others_R_model(more_features)
        R_feats = R_feats.view(
            batch_size, self.n_master2other_pairs, -1
        )
        Rs = torch.cat([
            self.master2others_R_param(R_feats[i]).unsqueeze(0)
            for i in range(batch_size)
        ])  # ~ batch_size, | others pairs |, (3 x 3)

        t_feats = self.master2others_t_model(more_features)
        t_feats = t_feats.view(
            batch_size, self.n_master2other_pairs, -1
        )
        ts = t_feats * self.scale_t

        return self.combiner(
            Rs,
            ts,
        ).view(batch_size, self.n_master2other_pairs, 4, 4)  # master 2 others

    def forward(self, x):
        """ batch ~ many poses, i.e ~ (batch_size, # views, n_joints, 2D) """

        features = self.backbone(x)  # batch_size, ...

        masters = self._forward_masters(features)
        master2others = self._forward_master2others(features)

        return torch.cat([
            masters.unsqueeze(1),
            master2others
        ], dim=1)
