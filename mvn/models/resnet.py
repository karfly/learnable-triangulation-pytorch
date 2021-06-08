from torch import nn


# todo refactor
class MLPResNet(nn.Module):
    def __init__(self, in_features, inner_size, n_inner_layers, out_features,
    batch_norm=False, drop_out=0.0, activation=nn.LeakyReLU, final_activation=None, init_weights=False):
        super().__init__()

        self.up = nn.Linear(in_features, inner_size, bias=True)
        sizes = (n_inner_layers + 1) * [inner_size]

        self.linears = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1], bias=True)
            for i in range(len(sizes) - 1)
        ])
        self.second_linears = nn.ModuleList([
            nn.Linear(sizes[i + 1], sizes[i + 1], bias=True)
            for i in range(len(sizes) - 1)
        ])

        self.bns = self._make_bn_layers(sizes, batch_norm)
        self.second_bns = self._make_bn_layers(sizes, batch_norm)

        # todo dropout
        self.activation = activation()

        self.head = nn.Linear(inner_size, out_features, bias=True)
        self.final_activation = final_activation() if (not final_activation is None) else None

        if init_weights:
            self._init_weights()

    def _make_bn_layers(self, sizes, batch_norm=True):
        return nn.ModuleList([
            nn.BatchNorm1d(sizes[i + 1], momentum=0.9) if batch_norm else None
            for i in range(len(sizes) - 1)
        ])

    def _init_weights(self):   # todo very stupid, can do better
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_layer(self, i, x, residual):
        l, b = self.linears[i], self.bns[i]
        l2, b2 = self.second_linears[i], self.second_bns[i]

        x = l(x)
        if not (b is None):
            x = b(x)
        x = self.activation(x)

        x = l2(x)
        if not (b2 is None):
            x = b2(x)

        x = x + residual
        x = self.activation(x)  # activation AFTER residual

        return x

    def _forward_final(self, x):
        x = self.head(x)
        if self.final_activation:
            x = self.final_activation(x)
        return x

    def forward(self, x):
        x = self.up(x)
        residual = x

        for i in range(len(self.linears)):
            x = self._forward_layer(i, x, residual)
            residual = x

        return self._forward_final(x)
