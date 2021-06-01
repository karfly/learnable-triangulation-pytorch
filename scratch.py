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

    kps = torch.tensor([[[ 3.8812e+00,  1.3926e+01],
         [ 3.3541e+00,  6.8653e+00],
         [ 2.5048e+00,  6.5423e-02],
         [-2.4629e+00, -6.4329e-02],
         [-2.7424e+00,  7.6849e+00],
         [-3.7910e+00,  1.5366e+01],
         [ 1.0069e-16, -3.4230e-16],
         [-1.6489e-01, -4.4771e+00],
         [-1.5538e-02, -9.0567e+00],
         [ 1.0274e+00, -1.2327e+01],
         [ 2.5402e+00, -5.1403e+00],
         [ 4.7449e+00, -3.1702e+00],
         [ 2.5662e+00, -7.9598e+00],
         [-2.8609e+00, -8.2904e+00],
         [-4.4612e+00, -3.6158e+00],
         [-5.6643e-01, -5.1077e+00],
         [ 5.6715e-01, -1.0342e+01]],

        [[-2.9508e+00,  1.2767e+01],
         [-2.7946e+00,  6.3119e+00],
         [-1.0288e+00, -7.7901e-01],
         [ 1.0738e+00,  8.1305e-01],
         [ 1.8091e+00,  8.1001e+00],
         [ 4.4583e+00,  1.3904e+01],
         [-4.5951e-16,  1.7564e-17],
         [ 2.6123e-01, -4.3563e+00],
         [-5.7645e-01, -8.4121e+00],
         [-3.2046e+00, -1.0964e+01],
         [-3.5514e+00, -4.3432e+00],
         [-2.2298e+00, -4.2528e+00],
         [-1.1329e+00, -8.2269e+00],
         [ 8.7585e-01, -7.0211e+00],
         [ 1.7112e+00, -1.8611e+00],
         [-2.0952e+00, -3.4499e+00],
         [-2.3318e+00, -9.1491e+00]],

        [[ 3.4908e+00,  1.5183e+01],
         [ 3.2223e+00,  7.7569e+00],
         [ 1.4100e+00,  4.3395e-01],
         [-1.3494e+00, -4.1530e-01],
         [-2.0532e+00,  7.1629e+00],
         [-4.6765e+00,  1.4199e+01],
         [ 5.8312e-16,  1.3665e-16],
         [-2.6784e-01, -4.5469e+00],
         [ 5.2162e-01, -9.0569e+00],
         [ 3.0211e+00, -1.2093e+01],
         [ 3.6886e+00, -4.6173e+00],
         [ 3.0212e+00, -2.5343e+00],
         [ 1.5197e+00, -7.7364e+00],
         [-1.2283e+00, -8.5838e+00],
         [-2.1698e+00, -4.1543e+00],
         [ 1.7296e+00, -4.9754e+00],
         [ 2.1766e+00, -1.0171e+01]],

        [[-4.8642e+00,  1.7298e+01],
         [-4.3047e+00,  9.0758e+00],
         [-3.0024e+00, -6.4774e-01],
         [ 3.0695e+00,  6.6221e-01],
         [ 3.4026e+00,  9.4832e+00],
         [ 4.6411e+00,  1.5421e+01],
         [-1.3810e-16, -5.5630e-16],
         [ 2.2762e-01, -5.6896e+00],
         [-5.9333e-02, -1.0819e+01],
         [-1.7622e+00, -1.3712e+01],
         [-3.6400e+00, -4.6303e+00],
         [-5.7174e+00, -4.8163e+00],
         [-3.1314e+00, -1.0401e+01],
         [ 3.6153e+00, -9.3623e+00],
         [ 5.6935e+00, -2.9562e+00],
         [ 4.3236e-01, -3.9035e+00],
         [-1.0334e+00, -1.1427e+01]]])
    viz_2ds(kps)
