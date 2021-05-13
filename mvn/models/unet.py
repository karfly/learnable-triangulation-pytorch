import torch
import torch.nn as nn

from mvn.models.layers import linear_with_activation


class MLUNet(nn.Module):
    """ https://arxiv.org/abs/1505.04597 with MLPs """

    def __init__(self, in_features, out_features, batch_norm=False, drop_out=0.0, activation=nn.LeakyReLU):  #todo use bn, drop
        super().__init__()

        self.encoder_0 = self._make_inner_block(in_features, 64, activation)
        self.encoder_1 = self._make_inner_block(64, 128, activation)
        self.encoder_2 = self._make_inner_block(128, 256, activation)
        self.encoder_3 = self._make_inner_block(256, 512, activation)

        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )

        self.decoder_2 = self._make_inner_block(256 + 512, 256, activation)
        self.decoder_1 = self._make_inner_block(128 + 256, 128, activation)
        self.decoder_0 = self._make_inner_block(128 + 64, 64, activation)

        self.head = nn.Linear(64, out_features)

    @staticmethod
    def _make_inner_block(in_features, out_features, activation):
        return nn.Sequential(*[
            nn.Flatten(),  # better be safe than sorry
            linear_with_activation(in_features, out_features, activation),
            linear_with_activation(out_features, out_features, activation)
        ])

    def encode(self, x):
        skip_conn_0 = self.encoder_0(x)  # save for later
        x = skip_conn_0  # self.maxpool

        skip_conn_1 = self.encoder_1(x)  # save for later
        x = skip_conn_1  # self.maxpool

        skip_conn_2 = self.encoder_2(x)  # save for later
        x = skip_conn_2  # self.maxpool

        return x, [skip_conn_0, skip_conn_1, skip_conn_2]

    def decode(self, x, skip_connections):
        # x = self.upsample(x)
        x = torch.cat([x, skip_connections[2]], dim=1)

        x = self.decoder_2(x)
        # x = self.upsample(x)
        x = torch.cat([x, skip_connections[1]], dim=1)

        x = self.decoder_1(x)
        # x = self.upsample(x)
        x = torch.cat([x, skip_connections[0]], dim=1)

        x = self.decoder_0(x)

        return x

    def forward(self, x):
        x, skip_connections = self.encode(x)  # down ...
        x = self.encoder_3(x)  # ... middle ...
        x = self.decode(x, skip_connections)  # ... up ...
        x = self.head(x)  # ... head

        return x