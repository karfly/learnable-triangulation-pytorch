from torch import nn


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class MLP(nn.Module):
    def __init__(self, sizes, batch_norm=False, activation=nn.LeakyReLU, init_weights=True):
        super().__init__()

        self.backbone = self._make_layers(sizes, batch_norm, activation)

    @staticmethod
    def _make_layers(sizes, batch_norm, activation):
        ls = []

        for i, size in enumerate(sizes):
            next_size = sizes[i + 1] if i + 1 < len(sizes) else None

            if next_size:
                ls.append(nn.Linear(size, next_size))

                if i + 2 < len(sizes):  # not on last layer
                    if batch_norm:
                        ls.append(nn.BatchNorm1d(next_size))

                    ls.append(activation(inplace=False))

        return nn.Sequential(*ls)

    def _initialize_weights(self):  # like VGG
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.backbone(x)


class ExtractEverythingLayer(nn.Module):
    def __init__(self, in_channels, mlp_sizes, batch_norm=False, init_weights=True):
        super().__init__()

        out_channels = mlp_sizes[0]
        self.conv_encoder = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=True,
            padding_mode='zeros'
        )

        inner_size = 128
        self.linear_encoder = nn.Sequential(
            nn.Linear(2 * in_channels * 2, inner_size),
            nn.LeakyReLU(inplace=False),

            nn.Linear(inner_size, inner_size),
            nn.LeakyReLU(inplace=False),

            nn.Linear(inner_size, inner_size),
            nn.LeakyReLU(inplace=False),

            nn.Linear(inner_size, out_channels)
        )

        mlp_sizes[0] *= 2
        self.mlp = MLP(
            mlp_sizes,
            batch_norm=batch_norm,
            activation=nn.LeakyReLU
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        batch_size = x.shape[0]

        x1 = self.conv_encoder(x).view(batch_size, -1)
        x2 = self.linear_encoder(x.view(batch_size, -1))
        x3 = torch.cat([
            x1, x2
        ], dim=1)

        return self.mlp(x3)

    def _initialize_weights(self):  # like VGG
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
