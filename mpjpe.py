import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # https://stackoverflow.com/a/56222305

from mvn.utils.tred import rotate_points, rotx, roty, rotz
from viz import draw_kps_in_3d


def generate_a_pose():
    return torch.tensor([
        [-282.0929,   41.2788,   93.4123],
        [-255.5418,   15.1562,  530.4614],
        [-167.1010,   82.4592,  968.7374],
        [  50.7919, -112.4910,  948.4948],
        [  84.9501,  -79.3025,  498.8638],
        [ 205.5588,   52.5867,   98.2917],
        [ -58.1541,  -15.0163,  958.6160],
        [ -45.0088,   -7.4817, 1219.3922],
        [ -75.4930,  -57.1768, 1463.5525],
        [-186.6068, -167.3549, 1619.1725],
        [-247.9671,  -92.0705, 1217.9148],
        [-272.4148,  143.3431, 1157.8699],
        [-169.8023,   76.7555, 1420.7063],
        [  54.4905, -153.1666, 1412.9734],
        [ 130.3678, -212.1822, 1139.4652],
        [-104.7593, -203.7575, 1204.7849],
        [-146.5972, -139.7094, 1514.9611]
    ]).float()


def generate_rotations(angle=90.0):
    rotations = [
        lambda theta: torch.mm(rotx(theta), roty(theta)),
        lambda theta: torch.mm(roty(theta), rotz(theta)),
        lambda theta: torch.mm(rotz(theta), rotx(theta)),
    ]

    theta = torch.tensor(np.deg2rad(angle))
    return torch.cat([
        r(theta).unsqueeze(0)
        for r in rotations
    ]), [
        '{:.0f}° around {} and {:.0f}° around {}'.format(angle, 'x', angle, 'y'),
        '{:.0f}° around {} and {:.0f}° around {}'.format(angle, 'y', angle, 'z'),
        '{:.0f}° around {} and {:.0f}° around {}'.format(angle, 'z', angle, 'x'),
    ]  # see `rotations`


def generate_poses():
    kps = generate_a_pose()
    rotations, labels = generate_rotations(angle=45)
    return kps, torch.cat([
        rotate_points(kps, r).unsqueeze(0)
        for r in rotations
    ]), labels


def calculate_mpjpe(a, b):
    return np.sqrt((
        (a - b) ** 2
    ).sum(1)).mean(0)


def plot_poses(original, poses, labels):
    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    draw_kps_in_3d(
        axis, original.detach().cpu().numpy(), label='original',
        marker='o', color='black'
    )

    for pose, label, color in zip(poses, labels, mcolors.TABLEAU_COLORS):
        label = '{}, MPJPE = {:.1f} mm'.format(
            label, calculate_mpjpe(original, pose)
        )
        draw_kps_in_3d(
            axis, pose.detach().cpu().numpy(),
            label=label, marker='o', color=color
        )

    axis.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


def main():
    original, poses, labels = generate_poses()
    plot_poses(original, poses, labels)


if __name__ == '__main__':
    main()
