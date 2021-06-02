import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # https://stackoverflow.com/a/56222305

from mvn.models.loss import GeodesicLoss
from mvn.utils.tred import rotz, rotation_matrix2axis_angle, euler_angles_to_matrix, matrix_to_euler_angles


def viz_geodesic():
    """ really appreciate https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html """

    def _gen_some_eulers():
        return np.float32([])

    rots = torch.cat([
        # rotx(torch.tensor(np.pi / 2)).unsqueeze(0),
        # roty(torch.tensor(np.pi / 3)).unsqueeze(0),
        rotz(torch.tensor(np.pi / 2)).unsqueeze(0),
        torch.tensor(R.random().as_matrix()).unsqueeze(0),
        torch.tensor(R.random().as_matrix()).unsqueeze(0),
        torch.tensor(R.random().as_matrix()).unsqueeze(0),
    ])

    distances = GeodesicLoss()._criterion(
        rots.float(),
        torch.eye(3, 3).repeat(rots.shape[0], 1, 1).float().to(rots.device)
    )

    # todo as separate f
    angle_axis = rotation_matrix2axis_angle(rots)

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    colors = plt.get_cmap('jet')(np.linspace(0, 1, rots.shape[0]))
    for aa, dist, color in zip(
        angle_axis.numpy(),
        distances.numpy(),
        colors):

        label = 'rotate by {:.1f} along [{:.1f}, {:.1f}, {:.1f}] => distance {:.1f}'.format(
            np.degrees(aa[-1]), aa[0], aa[1], aa[2], dist
        )
        axis.plot(
            [0, aa[0]],  # from origin ...
            [0, aa[1]],
            [0, aa[2]],  # ... to vec
            label=label,
            color=color,
        )

    # show axis
    axis.quiver(
        0, 0, 0,
        1, 0, 0,
        normalize=True,
        color='black',
    )
    axis.quiver(
        0, 0, 0,
        0, 1, 0,
        normalize=True,
        color='black',
    )
    axis.quiver(
        0, 0, 0,
        0, 0, 1,
        normalize=True,
        color='black',
    )

    axis.set_xlim3d(-2.0, 2.0)
    axis.set_ylim3d(-2.0, 2.0)
    axis.set_zlim3d(-2.0, 2.0)
    axis.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


def viz_se_smooth():
    def smooth(threshold, alpha, beta):
        def _f(x):
            x[x > threshold] = np.power(
                x[x > threshold],
                alpha
            ) * (threshold ** beta)  # soft version

            return x
        return _f

    n_points = 100
    xs = np.linspace(0, 1e3, n_points)

    threshold = 1e2

    _, axis = get_figa(1, 1, heigth=12, width=30)

    for alpha in np.linspace(0.1, 0.3, 2):
        for beta in np.linspace(0.9, 1.5, 3):
            ys = smooth(threshold, alpha, beta)(xs.copy())

            axis.plot(
                xs, ys,
                label='smoothed (alpha={:.1f}, beta={:.1f}'.format(alpha, beta)
            )

    axis.plot(xs, xs, label='MSE')

    axis.vlines(x=threshold, ymin=0, ymax=np.max(
        xs), linestyle=':', label='threshold')

    axis.set_xlim((xs[0], xs[-1]))
    axis.set_yscale('log')

    axis.legend(loc='upper left')


# todo refactor
def plot_vector(axis, vec, from_origin=True, color='black'):
    if from_origin:
        axis.quiver(
            0, 0, 0,
            *vec,
            normalize=True,
            color=color
        )
    else:
        axis.quiver(
            *vec,
            0, 0, 0,
            normalize=True,
            color=color
        )


def viz_extrinsics():
    convention = 'ZXY'
    eulers = torch.tensor([
        np.deg2rad(30), np.deg2rad(50), 0  # no pitch
    ])
    cam_orient_in_world = euler_angles_to_matrix(eulers, convention)
    aa_in_world = R.from_matrix(cam_orient_in_world.numpy()).as_rotvec()
    print('GT orient', np.rad2deg(eulers.numpy()))

    cam_R = cam_orient_in_world.T  # ext predicted
    aa_in_pose = R.from_matrix(cam_R).as_rotvec()
    print('eulers of predicted pose', np.rad2deg(
        matrix_to_euler_angles(cam_R, convention)
    ))

    print('eulers of predicted orientation', np.rad2deg(
        matrix_to_euler_angles(cam_R.T, convention)
    ))

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    plot_vector(axis, aa_in_world, from_origin=True, color='blue')
    plot_vector(axis, aa_in_pose, from_origin=True, color='red')

    plot_vector(axis, [1, 0, 0])  # X
    plot_vector(axis, [0, 1, 0])  # Y
    plot_vector(axis, [0, 0, 1])  # Z
    plt.show()


if __name__ == '__main__':
    # viz_geodesic()
    viz_extrinsics()
