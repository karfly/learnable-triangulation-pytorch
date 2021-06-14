from torch import nn
import torch

from mvn.models.skips import MLSkipper
from mvn.models.resnet import MLPResNet
from mvn.models.layers import R6DBlock, RodriguesBlock, DepthBlock, TranslationFromAnglesBlock


def make_mlp_by_name(name):
    if name == 'mlp':
        base_class = MLPResNet
    elif name == 'skip':
        base_class = MLSkipper

    def _f(in_features, inner_size, n_inner_layers, out_features, batch_norm, drop_out=0.0, activation=nn.LeakyReLU, final_activation=None, init_weights=False):
        return base_class(
            in_features=in_features,
            inner_size=inner_size,
            n_inner_layers=n_inner_layers,
            out_features=out_features,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
            final_activation=final_activation,
            init_weights=init_weights
        )

    return _f


class RotoTransCombiner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rotations, translations):
        batch_size = rotations.shape[0]
        n_views = rotations.shape[1]

        if translations.shape[-1] == 1:  # predicted just distance
            trans = torch.cat([  # ext.t in each view
                torch.zeros(batch_size, n_views, 2, 1).to(rotations.device),
                translations.unsqueeze(-1)  # massively helps to NOT use |.|
            ], dim=-2)  # vstack => ~ batch_size, | comparisons |, 3, 1
        else:
            trans = translations.unsqueeze(-1)

        roto_trans = torch.cat([  # ext (not padded) in each view
            rotations, trans
        ], dim=-1)  # hstack => ~ batch_size, | comparisons |, 3, 4
        roto_trans = torch.cat([  # padd each view
            roto_trans,
            torch.DoubleTensor(
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

        self.n_views_comparing = 4 + 24  # todo only if fake cams
        self.scale_t = config.cam2cam.postprocess.scale_t

        n_joints = config.model.backbone.num_joints
        batch_norm = config.cam2cam.model.batch_norm
        drop_out = config.cam2cam.model.drop_out
        n_features = config.cam2cam.model.master.n_features

        self.backbone = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            MLPResNet(
                in_features=self.n_views_comparing * n_joints * 2,
                inner_size=config.cam2cam.model.backbone.n_features,
                n_inner_layers=config.cam2cam.model.backbone.n_layers,
                out_features=n_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            # CAN be beneficial nn.BatchNorm1d(n_features),
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
            out_features=n_params_per_R * self.n_views_comparing,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=nn.LeakyReLU,
        )

        if config.cam2cam.data.pelvis_in_origin:
            self.t_model = DepthBlock(
                in_features=n_features,
                inner_size=n_features,
                n_inner_layers=config.cam2cam.model.master.t.n_layers,
                n2predict=self.n_views_comparing,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            )

    def _forward_R(self, features):
        R_feats = self.R_model(features)
        features_per_pair = R_feats.shape[-1] // self.n_views_comparing
        R_feats = R_feats.view(
            -1, self.n_views_comparing, features_per_pair
        )
        return torch.cat([  # ext.R in each view
            self.R_param(R_feats[batch_i]).unsqueeze(0)
            for batch_i in range(R_feats.shape[0])
        ])  # ~ batch_size, | n_predictions |, (3 x 3)

    def _forward_t(self, features):
        t_feats = self.t_model(features)  # ~ (batch_size, 3)
        trans = t_feats.view(-1, self.n_views_comparing, 1)
        return trans * self.scale_t

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
        self.n_master2other_pairs = self.n_views - 1  # 0 -> 1, 0 -> 2 ...
        self.scale_t = config.cam2cam.postprocess.scale_t

        n_joints = config.model.backbone.num_joints
        batch_norm = config.cam2cam.model.batch_norm
        drop_out = config.cam2cam.model.drop_out
        make_mlp_with = make_mlp_by_name(config.cam2cam.model.name)

        self.bb = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            make_mlp_with(
                in_features=self.n_views * n_joints * 2,
                inner_size=config.cam2cam.model.backbone.n_features,
                n_inner_layers=config.cam2cam.model.backbone.n_layers,
                out_features=config.cam2cam.model.master.n_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            # CAN be beneficial nn.BatchNorm1d(config.cam2cam.model.master.n_features),
        ])

        self.master_R, self.master_t = self._make_Rt_model(
            make_mlp_with,
            in_features=config.cam2cam.model.master.n_features,
            inner_size=config.cam2cam.model.master.n_features,
            R_param=config.cam2cam.model.master.R.parametrization,
            R_layers=config.cam2cam.model.master.R.n_layers,
            t_param=1 if config.cam2cam.data.pelvis_in_origin else 3,  # just d
            t_layers=config.cam2cam.model.master.t.n_layers,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=nn.LeakyReLU,
        )

        self.master2other_bb = nn.Sequential(*[
            nn.Flatten(),  # will be fed into a MLP
            make_mlp_with(
                in_features=4 * n_joints * 2,
                inner_size=config.cam2cam.model.backbone.n_features,
                n_inner_layers=config.cam2cam.model.backbone.n_layers,
                out_features=config.cam2cam.model.master.n_features,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            # CAN be beneficial nn.BatchNorm1d(config.cam2cam.model.master.n_features),
        ])

        self.master2other_R, _ = self._make_Rt_model(  # todo refactor
            make_mlp_with,
            in_features=config.cam2cam.model.master.n_features,
            inner_size=config.cam2cam.model.master.n_features,
            R_param=config.cam2cam.model.master2others.R.parametrization,
            R_layers=config.cam2cam.model.master2others.R.n_layers,
            t_param=3,
            t_layers=config.cam2cam.model.master2others.t.n_layers,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=nn.LeakyReLU,
        )

        self.master2other_t = nn.Sequential(*[
            make_mlp_with(
                in_features=config.cam2cam.model.master.n_features,
                inner_size=config.cam2cam.model.master.n_features,
                n_inner_layers=config.cam2cam.model.master2others.t.n_layers,
                out_features=2 * 4 + 1,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=nn.LeakyReLU,
            ),
            TranslationFromAnglesBlock()
        ])

    @staticmethod
    def _make_Rt_model(make_mlp, in_features, inner_size, R_param, R_layers, t_param, t_layers, batch_norm, drop_out, activation):
        if R_param == '6d':
            n_params_per_R = 6
            R_param = R6DBlock()
        elif R_param == 'rod':
            n_params_per_R = 3
            R_param = RodriguesBlock()

        R_model = nn.Sequential(*[
            make_mlp(
                in_features=in_features,
                inner_size=inner_size,
                n_inner_layers=R_layers,
                out_features=n_params_per_R,
                batch_norm=batch_norm,
                drop_out=drop_out,
                activation=activation
            ),
            R_param
        ])

        t_model = make_mlp(
            in_features=in_features,
            inner_size=inner_size,
            n_inner_layers=t_layers,
            out_features=t_param,
            batch_norm=batch_norm,
            drop_out=drop_out,
            activation=activation,
        )

        return R_model, t_model

    @staticmethod
    def _forward_cam(R_model, t_model, base_features, scale_t):
        Rs = R_model(base_features)  # ~ batch_size, (3 x 3)
        ts = t_model(base_features) * scale_t
        return RotoTransCombiner()(
            Rs.unsqueeze(1),
            ts.unsqueeze(1) if len(ts.shape) == 2 else ts,
        ).view(-1, 4, 4)

    def _forward_master(self, features):
        return self._forward_cam(
            self.master_R,
            self.master_t,
            features,
            self.scale_t
        )

    def _forward_master2others(self, x):
        features = self.master2other_bb(x)
        return torch.cat([
            self._forward_cam(
                self.master2other_R,
                self.master2other_t,
                features,
                self.scale_t
            ).unsqueeze(1)
            for _ in range(1, self.n_views)
        ], dim=1)

    def forward(self, x):
        """ batch ~ many poses, i.e ~ (batch_size, # views, n_joints, 2D) """

        features = self.bb(x)  # batch_size, ...
        masters = self._forward_master(features)
        master2others = self._forward_master2others(x)

        return torch.cat([
            masters.unsqueeze(1),  # batch_size, 4, 4 -> batch_size, 1, 4, 4
            master2others
        ], dim=1)
