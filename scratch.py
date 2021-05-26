import torch
import numpy as np
import matplotlib.colors as mcolors

from post.plots import get_figa
from viz import draw_kp_in_2d
from mvn.models.loss import KeypointsMSESmoothLoss


def viz_2ds(keypoints_2d):
    _, axis = get_figa(1, 1, heigth=10, width=5)
    colors = list(mcolors.TABLEAU_COLORS.values())

    for view_i, color in zip(range(keypoints_2d.shape[1]), colors):
        kps = keypoints_2d[0, view_i]
        norm = torch.norm(kps, p='fro') * 1e2

        label = 'view #{:0d} norm={:.2f}'.format(view_i, norm)
        draw_kp_in_2d(axis, kps.cpu().numpy(), label, color)

    axis.set_ylim(axis.get_ylim()[::-1])  # invert
    axis.legend(loc='lower right')


def viz_geodesic():
    # generate some (normalized) vectors

    # vec -> mat using Rodrigues formula

    # compute mat VS eye distance

    # show

    pass


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

    fig, axis = get_figa(1, 1, heigth=12, width=30)

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
