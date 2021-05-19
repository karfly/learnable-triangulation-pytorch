import torch
import torch.nn as nn


from mvn.models.resnet import MLPResNet


def linear_with_activation(in_features, out_features, activation):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        activation(inplace=False)  # better be safe than sorry
    )


class ResNetBlock(nn.Module):
    """ https://d2l.ai/chapter_convolutional-modern/resnet.html """

    def __init__(self, layer1, layer2, batch_norm=None, downsample=None, activation=nn.LeakyReLU, dropout=0, final_activation_before_residual=False):
        super().__init__()

        self.layer1 = layer1

        if batch_norm:
            self.bn1 = batch_norm[0]
        else:
            self.bn1 = None

        self.layer2 = layer2

        if batch_norm:
            self.bn2 = batch_norm[1]
        else:
            self.bn2 = None

        self.downsample = downsample

        try:
            self.activation = activation(inplace=False)
        except:
            self.activation = activation()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout, inplace=False)
        else:
            self.dropout = None

        self.final_activation_before_residual = final_activation_before_residual

    def forward(self, x):
        # 0: save residual
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x

        # 1: weight layer
        out = self.layer1(x)
        
        if self.bn1:
            out = self.bn1(out)

        out = self.activation(out)  # 2) activation

        if self.dropout:
            out = self.dropout(out)

        # 2: weight layer
        out = self.layer2(out)
        
        if self.bn2:
            out = self.bn2(out)

        # 3: final activation
        if self.final_activation_before_residual:
            out = self.activation(out)

            if self.dropout:
                out = self.dropout(out)

            out += residual  # 4: residual
        else:
            out += residual  # 4: residual
            out = self.activation(out)

        return out


class R6DBlock(nn.Module):
    """ https://arxiv.org/abs/1812.07035 """

    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize_vector(v, eps=1e-8):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))
        v_mag = torch.max(v_mag, torch.cuda.FloatTensor([eps]))
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        return v  # `nn.functional.normalize(v)`

    @staticmethod
    def cross_product(u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        return out

    def forward(self, x):
        x_raw = x[:, 0: 3]
        y_raw = x[:, 3: 6]

        x = nn.functional.normalize(x_raw)  # self.normalize_vector(x_raw)
        z = self.cross_product(x, y_raw)
        z = nn.functional.normalize(z)  # self.normalize_vector(z)

        y = self.cross_product(z, x)
        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)

        return torch.cat((x, y, z), 2)  # 3 x 3


# modified version of https://arxiv.org/abs/1709.01507, suitable for MLP
class SEBlock(nn.Module):
    def __init__(self, in_features, inner_size):
        super().__init__()

        # self.excite = nn.Sequential(*[
        #     nn.Linear(in_features, inner_size, bias=True),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(inner_size, in_features, bias=True),
        #     nn.Sigmoid(),
        # ])

        self.excite = nn.Sequential(*[
            MLPResNet(
                in_features,
                inner_size,
                2,
                in_features,
                batch_norm=True,
                init_weights=True
            ),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        # it's already squeezed ...
        activation_map = self.excite(x)  # excite
        return torch.mul(
            activation_map,
            x
        )  # attend
