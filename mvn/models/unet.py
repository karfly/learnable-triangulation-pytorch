import torch.nn as nn


class MLUNet(nn.Module):
    """ https://arxiv.org/abs/1505.04597 with MLPs """

    def __init__(self, in_features, out_features, drop_out=0.0, activation=nn.LeakyReLU):
        super().__init__()

        self.encoder_0 = self._make_inner_block(in_features, 64, activation)
        self.encoder_1 = self._make_inner_block(64, 128, activation)
        self.encoder_2 = self._make_inner_block(128, 256, activation)

        self.middle = self._make_inner_block(256, 512, activation)

        self.decoder_0 = self._make_inner_block(512, 256, activation)
        self.after_0 = self._make_after_skip_conn_block(256, activation)
        self.decoder_1 = self._make_inner_block(256, 128, activation)
        self.after_1 = self._make_after_skip_conn_block(128, activation)
        self.decoder_2 = self._make_inner_block(128, 64, activation)
        self.after_2 = self._make_after_skip_conn_block(64, activation)

        self.head = nn.Linear(64, out_features, bias=True)

    @staticmethod
    def _make_inner_block(in_features, out_features, activation):
        mid_features = (in_features + out_features) // 2  # gradually upscale

        return nn.Sequential(*[
            nn.Flatten(),  # better be safe than sorry
            
            nn.Linear(in_features, mid_features, bias=True),
            nn.BatchNorm1d(mid_features),
            activation(inplace=False),
            
            nn.Linear(mid_features, out_features, bias=True),
            nn.BatchNorm1d(out_features),
            activation(inplace=False),
        ])

    @staticmethod
    def _make_after_skip_conn_block(n_features, activation):
        return nn.Sequential(*[
            nn.BatchNorm1d(n_features),
            activation(inplace=False),
        ])

    def forward(self, x):  # I'm afraid putting x to list will remove grad
        # down ...
        skip_conn_0 = self.encoder_0(x)  # save for later
        skip_conn_1 = self.encoder_1(skip_conn_0)  # save for later
        skip_conn_2 = self.encoder_2(skip_conn_1)  # save for later

        x = self.middle(skip_conn_2)  # ... middle ...

        # ... up ...
        x = self.decoder_0(x)
        x = self.after_0(x + skip_conn_2)

        x = self.decoder_1(x)
        x = self.after_1(x + skip_conn_1)

        x = self.decoder_2(x)
        x = self.after_2(x + skip_conn_0)

        return self.head(x)  # ... head
