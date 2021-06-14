import numpy as np


def parse_epochs(epochs):
    loss_keys = filter(
        lambda x: 'loss / batch' in x,
        epochs[0].keys()
    )
    losses = {}

    for key in loss_keys:
        losses[key] = np.float64([
            np.mean(epoch[key])
            for epoch in epochs
        ])

    return losses
