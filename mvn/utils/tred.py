import torch


def find_plane_between(points):
    """ https://math.stackexchange.com/a/2306029/83083 """

    n_points = points.shape[0]
    dev = points.device

    A = torch.cat([
        torch.cat([
            points[point_i, 0].unsqueeze(0),
            points[point_i, 1].unsqueeze(0),
            torch.tensor(1.0).unsqueeze(0).to(dev)
        ]).unsqueeze(0)
        for point_i in range(n_points)
    ])  # xs, ys, 1
    b = torch.cat([
        torch.cat([
            points[point_i, 2].unsqueeze(0)
        ]).unsqueeze(0)
        for point_i in range(n_points)
    ])  # zs
    
    fit = torch.mm(
        torch.mm(
            torch.inverse(torch.mm(A.T, A)),
            A.T
        ),
        b
    )
    errors = b - torch.mm(A, fit)
    residual = torch.norm(errors)
    return fit, errors, residual
