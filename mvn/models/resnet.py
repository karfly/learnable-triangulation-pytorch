from torch import nn

from mvn.models.layers import ResNetBlock


class MLPResNetBlock(nn.Module):
    def __init__(self, n_layers, in_features, n_units, out_features, batch_norm=False, mlp=nn.Linear, activation=nn.LeakyReLU):
        super().__init__()

        layers = [
            mlp(in_features, n_units),
            activation
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

    def __init__(self, in_features, out_features, n_units=1024, n_blocks=2, mlp=nn.Linear, activation=nn.ReLU, batch_norm=True, dropout=0.5):
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
