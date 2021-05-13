from torch import nn


class MLP(nn.Module):
    def __init__(self, sizes, batch_norm=False, drop_out=0.0, linear=nn.Linear, activation=nn.LeakyReLU, init_weights=True):
        super().__init__()

        self.backbone = self._make_layers(sizes, batch_norm, drop_out, linear, activation)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layers(sizes, batch_norm, drop_out, linear, activation):
        ls = []

        for i, size in enumerate(sizes):
            next_size = sizes[i + 1] if i + 1 < len(sizes) else None

            if next_size:
                ls.append(linear(size, next_size))

                if i + 2 < len(sizes):  # not on last layer
                    if drop_out > 0.0:
                        ls.append(nn.Dropout(p=drop_out, inplace=False))

                    if batch_norm:
                        ls.append(nn.BatchNorm1d(next_size))

                    ls.append(activation(inplace=False))

        return nn.Sequential(*ls)

    def forward(self, x):
        return self.backbone(x)
