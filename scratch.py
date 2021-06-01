import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.transform import Rotation as R

from post.plots import get_figa
from viz import draw_kps_in_2d
from mvn.models.loss import GeodesicLoss
from mvn.utils.tred import rotx, roty, rotz, rotation_matrix2axis_angle


def viz_2ds(keypoints_2d):
    _, axis = get_figa(1, 1, heigth=10, width=5)
    colors = list(mcolors.TABLEAU_COLORS.values())

    for view_i, color in zip(range(keypoints_2d.shape[0]), colors):
        kps = keypoints_2d[view_i]
        norm = torch.norm(kps, p='fro') * 1e2

        label = 'view #{:0d} norm={:.2f}'.format(view_i, norm)
        draw_kps_in_2d(axis, kps.cpu().numpy(), label=label, color=color)

    axis.set_ylim(axis.get_ylim()[::-1])  # invert
    # axis.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()


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


if __name__ == '__main__':
    # viz_geodesic()

    kps = torch.tensor([[[ 5.3301e+00,  1.2160e+01],
         [ 1.2101e+00,  6.6978e+00],
         [-1.0309e+00, -5.8820e-01],
         [ 1.0758e+00,  6.1382e-01],
         [-1.0544e+00,  8.0417e+00],
         [-2.4664e+00,  1.4932e+01],
         [ 8.9719e-16, -4.9133e-16],
         [ 2.3679e-01, -4.0401e+00],
         [-2.2155e-01, -8.6411e+00],
         [-6.7964e-01, -1.1445e+01],
         [-3.2243e+00, -3.0472e+00],
         [-2.2484e+00, -3.3726e+00],
         [-1.3586e+00, -7.7048e+00],
         [ 1.5124e+00, -8.0546e+00],
         [ 4.6379e+00, -5.2429e+00],
         [ 2.5523e+00, -8.2240e+00],
         [-1.1762e+00, -9.6410e+00]],

        [[-5.7581e+00,  1.3348e+01],
         [-1.4507e+00,  7.5588e+00],
         [ 8.7368e-01,  7.7575e-01],
         [-8.3798e-01, -7.4405e-01],
         [ 1.0122e+00,  6.2608e+00],
         [ 2.1973e+00,  1.3032e+01],
         [ 3.4833e-16, -6.4752e-16],
         [-2.0115e-01, -3.6470e+00],
         [ 2.9487e-01, -7.9268e+00],
         [ 7.8364e-01, -1.0597e+01],
         [ 3.1902e+00, -3.1423e+00],
         [ 2.0958e+00, -1.1332e+00],
         [ 1.3027e+00, -6.1164e+00],
         [-1.2243e+00, -8.2548e+00],
         [-3.6751e+00, -7.3883e+00],
         [-1.5354e+00, -1.0855e+01],
         [ 1.3188e+00, -9.3754e+00]],

        [[ 3.3609e+00,  1.2815e+01],
         [ 1.7572e-01,  6.7973e+00],
         [-9.1042e-01, -6.3823e-01],
         [ 9.4955e-01,  6.6566e-01],
         [-2.2074e+00,  7.6221e+00],
         [-4.6090e+00,  1.4216e+01],
         [-4.1320e-16, -1.1577e-16],
         [ 8.2268e-01, -3.8779e+00],
         [ 1.0571e+00, -8.3900e+00],
         [ 1.0280e+00, -1.1160e+01],
         [-2.6528e+00, -3.4643e+00],
         [-1.6737e+00, -3.4723e+00],
         [-1.7545e-01, -7.5826e+00],
         [ 2.6385e+00, -7.6417e+00],
         [ 5.2339e+00, -4.6087e+00],
         [ 3.6651e+00, -7.8966e+00],
         [ 2.8889e-01, -9.5411e+00]],

        [[-3.9411e+00,  1.7450e+01],
         [-8.7897e-03,  9.5960e+00],
         [ 1.2976e+00,  9.6827e-01],
         [-1.2321e+00, -9.1940e-01],
         [ 2.6989e+00,  7.0548e+00],
         [ 5.6823e+00,  1.4761e+01],
         [ 6.6882e-17,  2.2233e-16],
         [-1.1256e+00, -4.3749e+00],
         [-1.5379e+00, -9.7867e+00],
         [-1.5694e+00, -1.3263e+01],
         [ 3.2600e+00, -4.6446e+00],
         [ 2.4082e+00, -1.5043e+00],
         [ 1.9556e-01, -7.6112e+00],
         [-3.5446e+00, -9.9733e+00],
         [-6.3834e+00, -8.5711e+00],
         [-4.5472e+00, -1.3505e+01],
         [-5.9073e-01, -1.1938e+01]]])
    viz_2ds(kps)
