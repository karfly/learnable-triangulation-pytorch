from torch import nn


def make_MLP(sizes=[64, 32, 16, 3], batch_norm=False, activation=nn.LeakyReLU):
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
