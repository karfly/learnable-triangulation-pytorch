import torch.nn as nn

def linear_with_activation(in_features, out_features, activation):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        activation(inplace=False)  # better be safe than sorry
    )