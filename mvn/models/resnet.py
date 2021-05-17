from torch import nn

linear = nn.Linear


class MLPResNet(nn.Module):
    def __init__(self, in_features, inner_size, n_inner_layers, out_features,
    batch_norm=False, drop_out=0.0, activation=nn.LeakyReLU, init_weights=False):
        super().__init__()

        sizes = (n_inner_layers + 1) * [inner_size]

        self.up = nn.Linear(in_features, inner_size, bias=True)

        self.linears = nn.ModuleList([
            linear(sizes[i], sizes[i + 1], bias=True)
            for i in range(len(sizes) - 1)
        ])
        self.second_linears = nn.ModuleList([
            linear(sizes[i + 1], sizes[i + 1], bias=True)
            for i in range(len(sizes) - 1)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sizes[i + 1]) if batch_norm else None
            for i in range(len(sizes) - 1)
        ])

        # todo dropout
        self.activation = activation(inplace=False)

        self.head = linear(inner_size, out_features, bias=True)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.up(x)
        residual = x

        for i in range(len(self.linears)):
            l, b = self.linears[i], self.bns[i]
            l2 = self.second_linears[i]

            x = l(x)
            if not (b is None):
                x = b(x)
            x = self.activation(x)

            x = l2(x)
            # no second batchnorm !!!

            x = x + residual
            x = self.activation(x)  # activation AFTER residual

            residual = x  # save for next layer

        x = self.head(x)
        return x
