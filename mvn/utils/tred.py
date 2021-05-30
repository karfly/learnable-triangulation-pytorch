import torch


def find_plane_minimizing_z(points):
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
    )  # ~ (3, 1)
    errors = b - torch.mm(A, fit)
    residual = torch.norm(errors)
    return fit, errors, residual


def perpendicular_distance(x1, y1, z1, a, b, c, d):
    d = torch.abs((a * x1 + b * y1 + c * z1 + d))
    e = torch.sqrt(a * a + b * b + c * c)
    return d / e


def find_plane_minimizing_normal(points):
    """ https://stackoverflow.com/a/53593404 """

    n_points = points.shape[0]
    dev = points.device

    # 1.calculate centroid of points and make points relative to it
    centroid = points.mean(axis=0)
    xyzR = points - centroid  # points relative to centroid

    # 2. calculate the singular value decomposition ...
    _, _, v = torch.svd(xyzR)
    normal = v[-1]  # ... and get the normal as the last column of v matrix
    normal = normal / torch.norm(normal)  # normalized

    # get d coefficient to plane
    d = normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2]

    errors = torch.cat([
        perpendicular_distance(  # todo faster
            points[point_i, 0], points[point_i, 1], points[point_i, 2],
            normal[0], normal[1], normal[2], d
        ).unsqueeze(0).to(dev)
        for point_i in range(n_points)
    ])
    residual = torch.norm(errors)

    fit = (normal[0], normal[1], normal[2], d)
    return fit, errors, residual
