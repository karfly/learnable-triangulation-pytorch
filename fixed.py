import numpy as np
import torch

from mvn.models.layers import RodriguesBlock
from mvn.utils.multiview import Camera, build_intrinsics, triangulate_batch_of_points, triangulate_point_from_multiple_views_linear_torch
from mvn.utils.img import rotation_matrix_from_vectors


def _get_random_R():
    """ random angle-axis -> Rodrigues """

    random_angle_axis = torch.tensor(np.random.rand(1, 3))
    return RodriguesBlock()(random_angle_axis).numpy()[0]


def _get_random_t():
    return np.random.normal(5e3, 1e2, size=3).T


def _get_random_z_t():
    return np.float64([
        0,
        0,
        np.random.normal(5e3, 1e2)
    ])


n_views = 3
keypoints_3d = torch.tensor(np.float64([
    np.random.normal(1e3, 1e2, size=3),  # a random one
    [0, 0, 0]  # origin
]))
print('GT world 3D KPs:')
print(keypoints_3d)

K = build_intrinsics(translation=(0, 0), f=(1e3, 1e3), shear=0)  # fixed for all

cameras = [
    Camera(_get_random_R(), _get_random_t(), K)
    for _ in range(n_views)
]

print('GT R|t:')
print(torch.tensor(np.float64([
    cam.extrinsics_padded
    for cam in cameras
])))

for cam in cameras:
    kp_in_cam = cam.world2cam()(keypoints_3d)
    pelvis_vector = kp_in_cam[1]

    # ... => find rotation matrix pelvis to z ...
    z_axis = [0, 0, 1]
    Rt = rotation_matrix_from_vectors(pelvis_vector, z_axis)

    # "At that point, after you re-sample, camera translation should be [0,0,d_pelvis]"
    cam.update_roto_extrsinsic(Rt)
print('Looking at pelvis (origin) ... => R|t ...')
print(torch.tensor(np.float64([
    cam.extrinsics_padded
    for cam in cameras
])))

print('... so now KPs in camspace are:')
print(torch.cat([
    cam.world2cam()(keypoints_3d).unsqueeze(0)
    for cam in cameras
]))

keypoints_2d = torch.cat([
    cam.world2proj()(keypoints_3d).unsqueeze(0)
    for cam in cameras
])
print('Working with KP projected:')
print(keypoints_2d)

print('... and simulating learning R|t. Learned R|t:')
learned_cams = [
    Camera(_get_random_R(), _get_random_z_t(), K)
    for _ in range(n_views)
]
print(torch.tensor(np.float64([
    cam.extrinsics_padded
    for cam in learned_cams
])))

proj_matricies = torch.cat([
    torch.tensor(cam.projection).unsqueeze(0)
    for cam in learned_cams
])
keypoints_3d = triangulate_batch_of_points(
    proj_matricies.unsqueeze(0).cpu(),
    keypoints_2d.unsqueeze(0).cpu(),
    triangulator=triangulate_point_from_multiple_views_linear_torch
)
print('DLT-ed KP 3D:')
print(keypoints_3d)
