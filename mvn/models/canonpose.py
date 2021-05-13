import torch.nn as nn

from mvn.models.layers import ResNetBlock

class CanonPose(nn.Module):
    """ https://arxiv.org/abs/2011.14679 """

    def __init__(self, in_features, out_features, inner_size=1024, batch_norm=False, dropout=0.0, activation=nn.LeakyReLU):
        super().__init__()

        self.fc = nn.Linear(in_features, inner_size)

        self.res_block_0 = self._make_res_block(
            inner_size, batch_norm, dropout, activation
        )

        # "each containing two consecutive residual blocks with identical architecture to the first block"
        n_inner_blocks = 2
        self.cam_path = nn.Sequential(*[
            self._make_res_block(inner_size, batch_norm, dropout, activation)
            for _ in range(n_inner_blocks)
        ])

        self.head = nn.Linear(inner_size, out_features)
    
    @staticmethod
    def _make_res_block(inner_size, batch_norm, dropout, activation):
        return ResNetBlock(
            nn.Linear(inner_size, inner_size),
            nn.Linear(inner_size, inner_size),
            batch_norm=[
                nn.BatchNorm1d(inner_size),
                nn.BatchNorm1d(inner_size),
            ] if batch_norm else None,
            activation=activation,
            dropout=dropout
        )

    def forward(self, x):
        x = self.fc(x)  # upscale "using a FC which then goes ..."
        x = self.res_block_0(x)  # "to a residual block"
        x = self.cam_path(x)  # "... 2 consecutive residual blocks ... "
        x = self.head(x)  # "... followed by a fully connected layer that downscales the features to the required size."

        return x