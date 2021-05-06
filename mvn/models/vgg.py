import torch
from torch import nn
from typing import Union, List, cast

# original configs
VGG_CONFIGS = {
    'smallest': [
        64, 'M',
    ],
    '11': [
        64, 'M',
        128, 'M',
        256, 256, 'M',
        512, 512, 'M',
        512, 512, 'M'
    ],
    '16': [
        64, 64, 'M',
        128, 128, 'M',
        256, 256, 256, 'M',
        512, 512, 512, 'M',
        512, 512, 512, 'M'
    ]
}


def make_layers(cfg, batch_norm=False, in_channels=3, kernel_size=3, activation=nn.ReLU):
    layers: List[nn.Module] = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), activation(inplace=False)]
            else:
                layers += [conv2d, activation(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        classifier: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = classifier
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_virgin_vgg(vgg_type, batch_norm=False, in_channels=3, kernel_size=3, num_classes=128, activation=nn.LeakyReLU, init_weights=True):
    cfg = VGG_CONFIGS[vgg_type]

    features = make_layers(
        cfg,
        batch_norm=batch_norm,
        in_channels=in_channels,
        kernel_size=kernel_size,
        activation=activation,
    )

    inner_size = 4096
    classifier = nn.Sequential(
        nn.Linear(int(cfg[-2]) * 7 * 7, inner_size),
        activation(inplace=False),
        nn.Dropout(inplace=False),

        nn.Linear(inner_size, inner_size),
        activation(inplace=False),
        nn.Dropout(inplace=False),

        nn.Linear(inner_size, num_classes),
    )
    return VGG(
        features, classifier, num_classes=num_classes, init_weights=init_weights
    )
