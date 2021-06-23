from torch import nn
import torch

from mvn.models.resnet import MLPResNet
from mvn.models.layers import R6DBlock, RodriguesBlock, DepthBlock, View


class RotoTransCombiner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rotations, translations):
        batch_size = rotations.shape[0]
        n_views = rotations.shape[1]

        if translations.shape[-2] == 1:  # predicted just distance
            trans = torch.cat([  # ext.t in each view
                torch.zeros(batch_size, n_views, 2, 1).to(translations.device),
                translations  # massively helps to NOT use |.|
            ], dim=-2)  # vstack => ~ batch_size, | comparisons |, 3, 1
        else:
            trans = translations  # alias

        roto_trans = torch.cat([  # ext (not padded) in each view
            rotations, trans
        ], dim=-1)  # hstack => ~ batch_size, | comparisons |, 3, 4
        roto_trans = torch.cat([  # padd each view
            roto_trans,
            torch.tensor(
                [0, 0, 0, 1]
            ).repeat(batch_size, n_views, 1, 1).to(roto_trans.device)
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

        self.n_views = 4
        if config.cam2cam.cams.use_extra_cams > 0:
            self.n_views += config.cam2cam.cams.use_extra_cams
        elif config.cam2cam.cams.using_just_one_gt:
            self.n_views = 3  # ALL but first

        self.scale_t = config.cam2cam.postprocess.scale_t

        n_joints = config.model.backbone.num_joints
        batch_norm = config.cam2cam.model.batch_norm
        drop_out = config.cam2cam.model.drop_out
        n_features = config.cam2cam.model.master.n_features
        activation = nn.LeakyReLU

        self.backbone = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                in_features=self.n_views * n_joints * 2,
                inner_size=config.cam2cam.model.backbone.n_features,
                n_inner_layers=config.cam2cam.model.backbone.n_layers,
                out_features=n_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
            ),
        ])

        n_params_per_R, self.R_param = None, None
        if config.cam2cam.model.master.R.parametrization == '6d':
            n_params_per_R = 6
            self.R_param = R6DBlock()
        elif config.cam2cam.model.master.R.parametrization == 'rod':
            n_params_per_R = 3
            self.R_param = RodriguesBlock()
        
        self.R_model = MLPResNet(
            in_features=n_features,
            inner_size=n_features,
            n_inner_layers=config.cam2cam.model.master.R.n_layers,
            out_features=n_params_per_R * self.n_views,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
        )

        if config.cam2cam.data.look_at_pelvis:  # just d
            self.t_model = DepthBlock(
                how_many=self.n_views,
                in_features=n_features,
                inner_size=n_features,
                n_inner_layers=config.cam2cam.model.master.t.n_layers,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
            )
        else:
            self.t_model = MLPResNet(
                in_features=n_features,
                inner_size=n_features,
                n_inner_layers=config.cam2cam.model.master.t.n_layers,
                out_features=3 * self.n_views,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation
            )

    def _forward_R(self, features):
        R_feats = self.R_model(features)
        features_per_pair = R_feats.shape[-1] // self.n_views
        R_feats = R_feats.view(
            -1, self.n_views, features_per_pair
        )
        return torch.cat([  # ext.R in each view
            self.R_param(R_feats[batch_i]).unsqueeze(0)
            for batch_i in range(R_feats.shape[0])
        ])  # ~ batch_size, | n_predictions |, (3 x 3)

    def _forward_t(self, features):
        feats = self.t_model(features)
        ds = feats.view(-1, self.n_views, feats.shape[-1] // self.n_views)
        return ds * self.scale_t

    def forward(self, x):
        """ batch ~ many poses, i.e ~ (batch_size, pair => 2, n_joints, 2D) """

        features = self.backbone(x)
        return RotoTransCombiner()(
            self._forward_R(features),
            self._forward_t(features)
        )


class Cam2camNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_views = 4
        self.n_others = self.n_views - 1  # 0 -> 1, 0 -> 2 ...
        self.scale_t = config.cam2cam.postprocess.scale_t

        n_joints = config.model.backbone.num_joints
        batch_norm = config.cam2cam.model.batch_norm
        drop_out = config.cam2cam.model.drop_out
        activation = nn.LeakyReLU

        self.bb = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                in_features=self.n_views * n_joints * 2,
                inner_size=config.cam2cam.model.backbone.n_features,
                n_inner_layers=config.cam2cam.model.backbone.n_layers,
                out_features=config.cam2cam.model.master.n_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
                final_activation=None,
                init_weights=False
            ),
        ])

        self.master_feats = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                in_features=1 * n_joints * 2,
                inner_size=config.cam2cam.model.master.n_features,
                n_inner_layers=3,
                out_features=config.cam2cam.model.master.n_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
                final_activation=None,
                init_weights=False
            )
        ])
        self.master_R = self.make_R_model(
            how_many=1,
            in_features=config.cam2cam.model.master.n_features,
            inner_size=config.cam2cam.model.master.n_features,
            param=config.cam2cam.model.master.R.parametrization,
            n_inner_layers=config.cam2cam.model.master.R.n_layers,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
        )
        self.master_t = DepthBlock(
            how_many=1,
            in_features=config.cam2cam.model.master.n_features,
            inner_size=config.cam2cam.model.master.n_features,
            n_inner_layers=config.cam2cam.model.master.t.n_layers,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
        )

        self.others_feats = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                in_features=self.n_others * n_joints * 2,
                inner_size=config.cam2cam.model.master.n_features,
                n_inner_layers=3,
                out_features=config.cam2cam.model.master.n_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
                final_activation=None,
                init_weights=False
            )
        ])
        self.others_R = self.make_R_model(
            how_many=self.n_others,
            in_features=config.cam2cam.model.master.n_features,
            inner_size=config.cam2cam.model.master.n_features,
            param=config.cam2cam.model.master2others.R.parametrization,
            n_inner_layers=config.cam2cam.model.master2others.R.n_layers,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
        )
        self.others_t = nn.Sequential(*[
            MLPResNet(
                in_features=config.cam2cam.model.master.n_features,
                inner_size=config.cam2cam.model.master.n_features,
                n_inner_layers=config.cam2cam.model.master2others.t.n_layers,
                out_features=self.n_others * 3,  # 3D euclidean space
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
                final_activation=None,
                init_weights=False
            ),
            View((-1, self.n_others, 3, 1))
        ])

    @staticmethod
    def make_R_model(how_many, in_features, inner_size, param, n_inner_layers, batch_norm, drop_out, activation):
        """ `how_many` refers to a per batch! """

        if param == '6d':
            n_params_per_R = 6
            R_param = R6DBlock()
        elif param == 'rod':
            n_params_per_R = 3
            R_param = RodriguesBlock()

        return nn.Sequential(*[
            MLPResNet(
                in_features=in_features,
                inner_size=inner_size,
                n_inner_layers=n_inner_layers,
                out_features=how_many * n_params_per_R,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation,
                final_activation=None,
                init_weights=False
            ),
            View((-1, n_params_per_R)),
            R_param,
            View((-1, how_many, 3, 3)),
        ])

    @staticmethod
    def _fix_prediction_shape(x):
        while len(x.shape) < 4:  # batch x samples x [..., ...]
            x = x.unsqueeze(1)

        return x

    def _forward_cam(self, R_model, t_model, features, scale_t):
        Rs = R_model(features)  # ~ batch_size, (3 x 3)
        ts = t_model(features) * scale_t
        return RotoTransCombiner()(
            self._fix_prediction_shape(Rs),
            self._fix_prediction_shape(ts),
        ).view(-1, 4, 4)

    def _forward_master(self, x, features):
        skip_feats = self.master_feats(x[:, 0])
        return self._forward_cam(
            self.master_R,
            self.master_t,
            skip_feats + features,
            self.scale_t
        ).view(-1, 1, 4, 4)

    def _forward_master2others(self, x, features):
        skip_feats = self.others_feats(x[:, 1:])
        return self._forward_cam(
            self.others_R,
            self.others_t,
            skip_feats + features,
            self.scale_t
        ).view(-1, self.n_others, 4, 4)

    def forward(self, x):
        """ batch ~ many poses, i.e ~ (batch_size, # views, n_joints, 2D) """

        features = self.bb(x)
        masters = self._forward_master(x, features)
        master2others = self._forward_master2others(x, features)
        return torch.cat([
            masters,
            master2others
        ], dim=1)
