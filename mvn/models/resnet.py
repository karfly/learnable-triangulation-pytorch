from torch import nn


activation = lambda: nn.LeakyReLU(negative_slope=1e-2, inplace=False)


class MLPResNet(nn.Module):
    def __init__(self, in_features, inner_size, n_inner_layers, out_features,
    batch_norm=False, drop_out=0.0, activation=activation, final_activation=None, init_weights=True):
        super().__init__()

        sizes = (n_inner_layers + 1) * [inner_size]

        self.up = nn.Linear(in_features, inner_size, bias=True)

        self.linears = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1], bias=True)
            for i in range(len(sizes) - 1)
        ])
        self.second_linears = nn.ModuleList([
            nn.Linear(sizes[i + 1], sizes[i + 1], bias=True)
            for i in range(len(sizes) - 1)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sizes[i + 1]) if batch_norm else None
            for i in range(len(sizes) - 1)
        ])

        # todo dropout
        self.activation = activation()

        self.head = nn.Linear(inner_size, out_features, bias=True)
        self.final_activation = final_activation

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    # init weights -> NaN
                    nn.init.constant_(m.bias, 0)  # no bias

                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)  # all should contribute ...
                    nn.init.constant_(m.bias, 0)  # ... no bias

    def forward(self, x):
        x = self.up(x)
        residual = x

        for i in range(len(self.linears)):
            l, b = self.linears[i], self.bns[i]

            x = l(x)
            if not (b is None):
                x = b(x)
            x = self.activation(x)

            l2 = self.second_linears[i]
            x = l2(x)

            x = x + residual
            x = self.activation(x)

            residual = x  # save for next layer

        x = self.head(x)

        if self.final_activation:
            x = self.final_activation(x)

        return x
