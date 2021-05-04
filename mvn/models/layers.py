from torch import nn


def make_MLP(sizes=[64, 32, 16, 3], batch_norm=False, activation=nn.LeakyReLU):
    ls = []

    for i, size in enumerate(sizes):
        next_size = sizes[i + 1] if i + 1 < len(sizes) else None

        if next_size:
            ls.append(nn.Linear(size, next_size))

            if batch_norm:
                ls.append(nn.BatchNorm1d(next_size))

            if i + 2 < len(sizes):  # not on last layer
                ls.append(activation(inplace=True))

    return nn.Sequential(*ls)
