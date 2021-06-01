import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from functools import reduce
from mvn.utils.img import rotation_matrix_from_vectors_torch


# todo refactor everything diocan


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


def find_plane_minimizing_normal(points):
    """ https://www.ltu.se/cms_fs/1.51590!/svd-fitting.pdf """

    centroid = points.mean(axis=0)
    points_centered = points - centroid
    u, _, v = torch.svd(points_centered)
    normal = u[2, :]  # normal vector of the best-fitting plane

    d = - torch.dot(centroid, normal)  # distance from origin
    fit = torch.cat(
        [normal, d.unsqueeze(0)], axis=0
    )
    return fit, None, None  # todo errors, residual


def perpendicular_distance(x1, y1, z1, a, b, c, d):
    d = torch.abs((a * x1 + b * y1 + c * z1 + d))
    e = torch.sqrt(a * a + b * b + c * c)
    return d / e


def project_point_on_line(a, b, p):
    """ a: a point of the line
        b: the other point of the line
        p: the point you want to project
    """

    ap = p - a
    ab = b - a
    return a + torch.dot(ap, ab) / torch.dot(ab, ab) * ab


def distance_point_2_line(a, b, p):  # todo faster
    projected = project_point_on_line(a, b, p)
    return torch.norm(
        p - projected,
        p='fro'
    )


def find_line_minimizing_normal(points):
    """ https://scikit-spatial.readthedocs.io/en/stable/api_reference/Line/methods/skspatial.objects.Line.best_fit.html#skspatial.objects.Line.best_fit """

    n_points = points.shape[0]
    dev = points.device

    centroid = points.mean(axis=0)
    points_centered = points - centroid
    u, _, v = torch.svd(points_centered)
    direction = u[2, :]  # line is parametrized as `centroid + t * direction`

    a = centroid
    b = centroid + direction
    errors = torch.cat([
        distance_point_2_line(a, b, points[point_i]).unsqueeze(0).to(dev)
        for point_i in range(n_points)
    ])
    residual = torch.norm(errors)

    fit = (centroid, direction)
    return fit, errors, residual


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """ https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/transforms/rotation_conversions.py """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix, convention: str):
    """ https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.matrix_to_euler_angles """

    def _index_from_letter(letter: str):
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2

    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def _axis_angle_rotation(axis: str, angle):
    """ https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """ https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.euler_angles_to_matrix """

    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return reduce(torch.matmul, matrices)


def rotx(theta):
    """ theta rotation around x axis """

    return torch.DoubleTensor([
        [ 1, 0, 0],
        [ 0, torch.cos(theta), -torch.sin(theta)],
        [0, torch.sin(theta), torch.cos(theta)]
    ])


def roty(theta):
    """ theta rotation around y axis """

    return torch.DoubleTensor([
        [ torch.cos(theta), 0, torch.sin(theta)],
        [ 0, 1, 0],
        [ -torch.sin(theta), 0, torch.cos(theta)]
    ])


def rotz(theta):
    """ theta rotation around z axis """

    return torch.DoubleTensor([
        [ torch.cos(theta), -torch.sin(theta), 0],
        [ torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1]
    ])


def rotate_points(points, R):
    return torch.mm(
        points,
        R.T.type(points.dtype)
    )  # R * points ...


# todo separate f
def _rotate_points_based_on_joint_align(points, ref_points, joint_i):
    return rotate_points(
        points,
        rotation_matrix_from_vectors_torch(
            points[joint_i],
            ref_points[joint_i]
        )
    )


def create_plane(C):
    a, b, c, d = C  # unpack
    x, y = np.meshgrid(
        np.arange(-2e2, 2e2, 10),
        np.arange(-2e2, 2e2, 10)
    )
    z = (d - (a*x + b*y)) / c
    # matrix version: Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
    return x, y, z


def create_line(coefficients):
    centroid, direction = coefficients  # unpack
    ts = torch.arange(-2e2, 2e2, 10)
    return torch.cat([
        torch.DoubleTensor([
            centroid[0] + t * direction[0],
            centroid[1] + t * direction[1],
            centroid[2] + t * direction[2],
        ]).unsqueeze(0)
        for t in ts
    ])


# warning, grad may not work
def rotation_matrix2axis_angle(batch_rotations):
    def _mat2aa(m):
        axis = torch.tensor(R.from_matrix(m).as_rotvec())
        axis = axis / torch.norm(axis, p='fro')  # normalize

        angle = torch.tensor(
            np.arccos((np.trace(m) - 1) / 2.0)
        )
        return torch.cat([
            axis, angle.unsqueeze(0)
        ], dim=0)

    return torch.cat([
        _mat2aa(rot).unsqueeze(0)
        for rot in batch_rotations
    ])


def mirror_points_along_z(z):
    def _f(points):
        out = points.clone()
        n_points = points.shape[0]
        for point_i in range(n_points):  # todo batched
            old_z = points[point_i, 2]
            if old_z < z:
                new_z = z + (z - old_z)  # todo simplify
            else:
                new_z = z - (old_z - z)
            
            out[point_i, 2] = new_z
        return out
    return _f


def mirror_points_along_y(y):
    def _f(points):
        out = points.clone()
        n_points = points.shape[0]
        for point_i in range(n_points):  # todo batched
            old_y = points[point_i, 1]
            if old_y < y:
                new_y = y + (y - old_y)  # todo simplify
            else:
                new_y = y - (old_y - y)
            
            out[point_i, 1] = new_y
        return out
    return _f


def mirror_points_along_x(x):
    def _f(points):
        out = points.clone()
        n_points = points.shape[0]
        for point_i in range(n_points):  # todo batched
            old_x = points[point_i, 0]
            if old_x < x:
                new_x = x + (x - old_x)  # todo simplify
            else:
                new_x = x - (old_x - x)
            
            out[point_i, 0] = new_x
        return out
    return _f


def get_cam_location_in_world(exts):
    return torch.inverse(exts)
