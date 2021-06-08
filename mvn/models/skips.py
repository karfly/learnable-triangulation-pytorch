from torch import nn
import torch


class MLSkipper(nn.Module):
    """ A MLP with skip connections (= symmetric residual connections) """

    def __init__(self, in_features, inner_size, n_inner_layers, out_features,
    batch_norm=False, drop_out=0.0, activation=nn.LeakyReLU, final_activation=None, init_weights=False):
        super().__init__()

        self.up = nn.Linear(in_features, inner_size, bias=True)

        self.encode_linears = nn.ModuleList([
            nn.Linear(inner_size, inner_size, bias=True)
            for _ in range(n_inner_layers)
        ])
        self.decode_linears = nn.ModuleList([
            nn.Linear(inner_size * 2, inner_size, bias=True)
            for _ in range(n_inner_layers)
        ])

        self.encode_bns = self._make_bn_layers(inner_size, n_inner_layers, batch_norm)
        self.decode_bns = self._make_bn_layers(inner_size, n_inner_layers, batch_norm)

        # todo dropout
        self.activation = activation()

        self.head = nn.Linear(inner_size, out_features, bias=True)
        self.final_activation = final_activation() if (not final_activation is None) else None

        if init_weights:
            self._init_weights()

    def _make_bn_layers(self, inner_size, n_inner_layers, batch_norm=True):
        return nn.ModuleList([
            nn.BatchNorm1d(inner_size) if batch_norm else None
            for _ in range(n_inner_layers)
        ])

    def _init_weights(self):   # todo very stupid, can do better
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)

    def _encode_layer(self, i, x):
        l, b = self.encode_linears[i], self.encode_bns[i]

        x = l(x)
        if not (b is None):
            x = b(x)
        x = self.activation(x)

        # todo more inner layers?

        return x

    def _forward_encode(self, x):
        skip_connections = []

        for i in range(len(self.encode_linears)):
            x = self._encode_layer(i, x)
            skip_connections.append(x)

        return x, skip_connections

    def _decode_layer(self, i, x, skip_connection):
        l, b = self.decode_linears[i], self.decode_bns[i]

        x = torch.cat([
            x, skip_connection
        ], dim=1)

        x = l(x)
        if not (b is None):
            x = b(x)
        x = self.activation(x)

        # todo more inner layers?

        return x

    def _forward_decode(self, x, skip_connections):
        for i in range(len(skip_connections)):
            x = self._decode_layer(i, x, skip_connections[-(i + 1)])

        return x

    def _forward_final(self, x):
        x = self.head(x)
        if self.final_activation:
            x = self.final_activation(x)
        return x

    def forward(self, x):
        x = self.up(x)
        x, skip_connections = self._forward_encode(x)
        x = self._forward_decode(x, skip_connections)
        return self._forward_final(x)
