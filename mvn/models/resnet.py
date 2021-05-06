from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, layer1, layer2, batch_norm=None, downsample=None, activation=nn.LeakyReLU):
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

    def forward(self, x):
        residual = x  # https://d2l.ai/chapter_convolutional-modern/resnet.html

        out = self.layer1(x)  # 1) weight
        if self.bn1:
            out = self.bn1(out)

        out = self.activation(out)  # 2) activation

        out = self.layer2(out)  # 3) weight
        if self.bn2:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(residual)

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
