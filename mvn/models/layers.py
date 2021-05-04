from torch import nn


def make_MLP(sizes=[64, 32, 16], n_classes=3, activation=nn.LeakyReLU):
    ls = []

    for i, size in enumerate(sizes):
        next_size = sizes[i + 1] if i + 1 < len(sizes) else None

        if next_size:
            ls.append(nn.Linear(size, next_size))
            ls.append(activation(inplace=True))
        else:  # output head
            last_size = sizes[-1]
            ls.append(nn.Linear(last_size, n_classes))

    return nn.Sequential(*ls)
