from torch import nn
from torch.nn.modules import activation


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

        self.activation = activation(inplace=False)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout, inplace=False)
        else:
            self.dropout = None

        self.final_activation_before_residual = final_activation_before_residual

    def forward(self, x):
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.layer1(x)  # 1) weight
        
        if self.bn1:
            out = self.bn1(out)

        out = self.activation(out)  # 2) activation

        if self.dropout:
            out = self.dropout(out)

        out = self.layer2(out)  # 3) weight
        
        if self.bn2:
            out = self.bn2(out)

        if self.final_activation_before_residual:
            out = self.activation(out)  # 4) activation

            if self.dropout:
                out = self.dropout(out)

            out += residual  # 5) residual
        else:
            out += residual  # 4) residual
            out = self.activation(out)  # 5) activation

        return out


class MLPResNetBlock(nn.Module):
    def __init__(self, n_layers, in_features, n_units, out_features, batch_norm=False, mlp=nn.Linear, activation=nn.LeakyReLU):
        super().__init__()

        layers = [
            mlp(in_features, n_units),
            activation(inplace=False)
        ]  # input
        layers += [
            ResNetBlock(
                mlp(n_units, n_units),
                mlp(n_units, n_units),
                batch_norm=[
                    nn.BatchNorm1d(n_units),
                    nn.BatchNorm1d(n_units),
                ] if batch_norm else None,
                activation=activation,
            )
            for _ in range(n_layers)
        ]  # bottleneck
        layers += [
            mlp(n_units, out_features)
        ]  # output

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MartiNet(nn.Module):
    """ https://arxiv.org/abs/1705.03098 """

    def __init__(self, in_features, out_features, n_units=1024, n_blocks=2, mlp=nn.Linear, batch_norm=True, activation=nn.ReLU, dropout=0.5):
        super().__init__()

        layers = [
            mlp(in_features, n_units),
            activation(inplace=False)
        ]  # input
        layers += [
            ResNetBlock(
                mlp(n_units, n_units),
                mlp(n_units, n_units),
                batch_norm=[
                    nn.BatchNorm1d(n_units),
                    nn.BatchNorm1d(n_units),
                ] if batch_norm else None,
                activation=activation,
                dropout=dropout,
                final_activation_before_residual=True
            )
            for _ in range(n_blocks)
        ]  # bottleneck
        layers += [
            mlp(n_units, out_features)
        ]  # output

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
