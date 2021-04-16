import numpy as np
import torch


class Camera:
    def __init__(self, R, t, K, dist=None, name=""):
        self.R = np.array(R).copy()
        assert self.R.shape == (3, 3)

        self.t = np.array(t).copy()
        assert self.t.size == 3
        self.t = self.t.reshape(3, 1)

        self.K = np.array(K).copy()
        assert self.K.shape == (3, 3)

        self.dist = dist
        if self.dist is not None:
            self.dist = np.array(self.dist).copy().flatten()

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])


def euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous

    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def project_3d_points_to_image_plane_without_distortion(proj_matrix, points_3d, convert_back_to_euclidean=True):
    """Project 3D points to image plane not taking into account distortion
    Args:
        proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
        points_3d numpy array or torch tensor of shape (N, 3): 3D points
        convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
    NOTE: division by zero can be here if z = 0
    Returns:
        numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
    """
    if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.T
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def triangulate_point_from_multiple_views_in_master_space(proj_matricies, points):
    pass  # todo


def triangulate_point_from_multiple_views_linear(proj_matricies, points):
    """Triangulates one point from multiple (N) views using direct linear transformation (DLT). For more information look at "Multiple view geometry in computer vision", Richard Hartley and Andrew Zisserman, 12.2 (p. 312).

    Args:
        proj_matricies numpy array of shape (N, 3, 4): sequence of projection matricies (3x4)
        points numpy array of shape (N, 2): sequence of points' coordinates

    Returns:
        point_3d numpy array of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)
    A = np.zeros((2 * n_views, 4))
    for j in range(len(proj_matricies)):
        A[j * 2 + 0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
        A[j * 2 + 1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]

    u, s, vh = np.linalg.svd(A, full_matrices=False)
    point_3d_homo = vh[3, :]

    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d


def triangulate_point_from_multiple_views_linear_torch(proj_matricies, points, confidences=None):
    """ = triangulate_point_from_multiple_views_linear for PyTorch.
    Args:
        proj_matricies torch tensor of shape (N, 3, 4): sequence of projection matricies (3x4), where N is the number of views
        points torch tensor of of shape (N, 2): sequence of points' coordinates
        confidences None or torch tensor of shape (N,): confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0
    Returns:
        point_3d numpy torch tensor of shape (3, ): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)

    if confidences is None:
        confidences = torch.ones(n_views, dtype=torch.float32, device=points.device)

    A = proj_matricies[:, 2:3].expand(n_views, 2, 4) * points.view(n_views, 2, 1)
    A -= proj_matricies[:, :2]
    A *= confidences.view(-1, 1, 1)

    u, s, vh = torch.svd(A.view(-1, 4))

    point_3d_homo = -vh[:, 3]
    point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]

    return point_3d


def triangulate_from_multiple_views_sii(proj_matricies, points, n_iter=2):
    """ from https://github.com/edoRemelli/DiffDLT/blob/master/dlt.py#L42. This module lifts batch_size 2d detections obtained from n_views viewpoints to 3D using the DLT method. It computes the eigenvector associated to the smallest eigenvalue using the Shifted Inverse Iterations algorithm.
    Args:
        proj_matricies torch tensor of shape (batch_size, n_views, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of shape (batch_size, n_views, 2): sequence of points' coordinates
    Returns:
        point_3d torch tensor of shape (batch_size, 3): triangulated points
    """

    batch_size = proj_matricies.shape[0]
    n_views = proj_matricies.shape[1]

    # assemble linear system
    A = proj_matricies[:, :, 2:3].expand(batch_size, n_views, 2, 4) * points.view(-1, n_views, 2, 1)
    A -= proj_matricies[:, :, :2]
    A = A.view(batch_size, -1, 4)

    AtA = A.permute(0, 2, 1).matmul(A).float()
    I = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1).to(A.device)
    B = AtA + 0.001 * I  # avoid numerical errors

    # initialize normalized random vector
    bk = torch.rand(batch_size, 4, 1).float().to(AtA.device)
    norm_bk = torch.sqrt(bk.permute(0, 2, 1).matmul(bk))
    bk = bk / norm_bk
    
    for k in range(n_iter):
        bk, _ = torch.solve(bk, B)
        norm_bk = torch.sqrt(bk.permute(0, 2, 1).matmul(bk))
        bk = bk / norm_bk

    point_3d_homo = -bk.squeeze(-1)
    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d


def triangulate_batch_of_points_using_gpu_friendly_svd(proj_matricies_batch, points_batch):
    batch_size, n_views, n_joints = points_batch.shape[:3]
    point_3d_batch = torch.zeros(
        batch_size,
        n_joints,
        3,  # because we're in 3D space
        dtype=torch.float32,
        device=points_batch.device
    )  # ~ (batch_size=8, n_joints=17, 3)

    for batch_i in range(batch_size):  # 8
        for joint_i in range(n_joints):  # 17
            points = points_batch[batch_i,:, joint_i,:]

            point_3d = triangulate_from_multiple_views_sii(
                proj_matricies_batch[batch_i].unsqueeze(0),  # ~ (n_views=4, 3, 4)
                points.unsqueeze(0)  # ~ (n_views=4, 2)
            )  # ~ (3, )
            point_3d_batch[batch_i, joint_i] = point_3d

    return point_3d_batch


def triangulate_batch_of_points(proj_matricies_batch, points_batch, confidences_batch=None):
    """ proj matrices, keypoints 2D (pred), confidences """

    batch_size, n_views, n_joints = points_batch.shape[:3]
    point_3d_batch = torch.zeros(
        batch_size,
        n_joints,
        3,
        dtype=torch.float32,
        device=points_batch.device
    )  # ~ (batch_size=8, n_joints=17, 3)

    for batch_i in range(batch_size):  # 8
        for joint_i in range(n_joints):  # 17
            points = points_batch[batch_i, :, joint_i, :]

            confidences = confidences_batch[batch_i, :, joint_i] if confidences_batch is not None else None
            point_3d = triangulate_point_from_multiple_views_linear_torch(
                proj_matricies_batch[batch_i],  # ~ (n_views=4, 3, 4)
                points,  # ~ (n_views=4, 2)
                confidences=confidences  # ~ (n_views=4, )
            )  # ~ (3, )
            point_3d_batch[batch_i, joint_i] = point_3d

    return point_3d_batch


def calc_reprojection_error_matrix(keypoints_3d, keypoints_2d_list, proj_matricies):
    reprojection_error_matrix = []
    for keypoints_2d, proj_matrix in zip(keypoints_2d_list, proj_matricies):
        keypoints_2d_projected = project_3d_points_to_image_plane_without_distortion(proj_matrix, keypoints_3d)
        reprojection_error = 1 / 2 * np.sqrt(np.sum((keypoints_2d - keypoints_2d_projected) ** 2, axis=1))
        reprojection_error_matrix.append(reprojection_error)

    return np.vstack(reprojection_error_matrix).T
