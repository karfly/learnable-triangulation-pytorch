from pathlib import Path
import torch
import numpy as np
import argparse

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # https://stackoverflow.com/a/56222305

from mvn.mini import get_config
from mvn.pipeline.setup import setup_dataloaders


def get_joints_connections():
    return [
        (4, 5),
        (6, 3),  # pelvis -> left anca
        (3, 4),  # left anca -> left knee
        (6, 7),  # pelvis -> back
        (6, 2),  # pelvis -> right anca
        (2, 1),  # right anca -> right knee
        (1, 0),  # right knee -> right foot
        (7, 8),
        (8, 9),
        (9, 16),
        (8, 13),
        (13, 14),
        (15, 14),
        (8, 12),
        (12, 11),
        (11, 10)
    ]


def draw_kp_in_2d(axis, keypoints_2d_in_view, label, color):
    for i, joint_pair in enumerate(get_joints_connections()):
        joints = [
            keypoints_2d_in_view[joint_pair[0]],
            keypoints_2d_in_view[joint_pair[1]]
        ]
        xs = joints[0][0], joints[1][0]
        ys = joints[0][1], joints[1][1]

        if i == 0:
            axis.plot(
                xs, ys,
                marker='o',
                markersize=10,
                color=color,
                label=label
            )
        else:
            axis.plot(
                xs, ys,
                marker='o',
                markersize=15,
                color=color,
            )


def draw_kp_in_3d(axis, keypoints_3d, label=None, marker='o', color='blue'):
    for joint_pair in get_joints_connections():
        joints = [
            keypoints_3d[joint_pair[0]],
            keypoints_3d[joint_pair[1]]
        ]
        xs = joints[0][0], joints[1][0]
        ys = joints[0][1], joints[1][1]
        zs = joints[0][2], joints[1][2]

        axis.plot(
            xs, ys, zs,
            marker=marker,
            markersize=0 if label else 5,
            color=color,
            # todo too many label=label,
        )

    if label:
        xs = keypoints_3d[:, 0]
        ys = keypoints_3d[:, 1]
        zs = keypoints_3d[:, 2]
        n_points = keypoints_3d.shape[0]

        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1, n_points))
        for point_i in range(n_points):
            axis.scatter(
                [ xs[point_i] ], [ ys[point_i] ], [ zs[point_i] ],
                marker='o',
                color=colors[point_i],
                label=label + ' {:.0f}'.format(point_i)
                # todo too many label=label,
            )



def load_data(config, dumps_folder):
    def _load(file_name):
        f_path = dumps_folder / file_name
        return torch.load(f_path).cpu().numpy()

    keypoints_3d_gt = _load('kps_world_gt.trc')  # see `cam2cam:_save_stuff`
    keypoints_3d_pred = _load('kps_world_pred.trc')

    indices = None  # _load('batch_indexes.trc')
    _, val_dataloader, _ = setup_dataloaders(config, distributed_train=False)  # ~ 0 seconds

    return keypoints_3d_gt, keypoints_3d_pred, indices, val_dataloader


def get_dump_folder(milestone, experiment):
    tesi_folder = Path('~/Scuola/now/thesis').expanduser()
    milestones = tesi_folder / 'milestones'
    current_milestone = milestones / milestone
    folder = 'human36m_alg_AlgebraicTriangulationNet@{}'.format(experiment)
    return current_milestone / folder / 'epoch-0-iter-0'


def main(config, milestone, experiment_name):
    dumps_folder = get_dump_folder(milestone, experiment_name)
    gts, pred, indices, dataloader = load_data(config, dumps_folder)

    per_pose_error_relative, per_pose_error_absolute, _ = dataloader.dataset.evaluate(
        pred,
        split_by_subject=True,
        keypoints_gt_provided=gts,
    )  # (average 3D MPJPE (relative to pelvis), all MPJPEs)

    message = 'MPJPE relative to pelvis: {:.1f} mm, absolute: {:.1f} mm'.format(
        per_pose_error_relative,
        per_pose_error_absolute
    )  # just a little bit of live debug
    print(message)

    max_plots = 6
    n_samples = gts.shape[0]
    n_plots = min(max_plots, n_samples)
    samples_to_show = np.random.permutation(np.arange(n_samples))[:n_plots]

    print('found {} samples but plotting {}'.format(n_samples, n_plots))

    fig = plt.figure(figsize=plt.figaspect(1.5))
    fig.set_facecolor('white')
    for i, sample_i in enumerate(samples_to_show):
        axis = fig.add_subplot(2, 3, i + 1, projection='3d')

        draw_kp_in_3d(axis, gts[sample_i], 'GT (resampled)', 'o', 'blue')
        draw_kp_in_3d(axis, pred[sample_i], 'prediction', '^', 'red')
        print(
            'sample #{} (#{}): pelvis predicted @ ({:.1f}, {:.1f}, {:.1f})'.format(
                i,
                sample_i,
                pred[sample_i, 6, 0],
                pred[sample_i, 6, 1],
                pred[sample_i, 6, 2],
            )
        )

        # axis.legend(loc='lower left')

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--milestone', type=str, required=True,
        help='milestone name, e.g "20.05_27.05_rodrigezzzzzzzzzz"'
    )
    parser.add_argument(
        '--exp', type=str, required=True,
        help='experiment name, e.g "25.05.2021-18:58:36")'
    )

    return parser.parse_args()


def create_plane(coefficients):
    X, Y = np.meshgrid(
        np.arange(-1e2, 1e2),
        np.arange(-1e2, 1e2)
    )
    Z = C[0] * X + C[1] * Y + C[2]
    # matrix version: Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
    return X, Y, Z


def debug():
    pred = torch.tensor([[-9.6090e+01,  3.4459e+02, -7.9170e+02],
        [-4.1182e+01,  1.4140e+02, -3.9417e+02],
        [-1.2413e+01, -6.1242e-01,  3.0517e+01],
        [ 9.1476e+00,  9.5254e-01, -2.6849e+01],
        [-1.7301e+01,  1.2722e+02, -4.2442e+02],
        [-1.0021e+02,  3.3884e+02, -8.2207e+02],
        [ 1.6836e-11,  2.4591e-11, -6.8228e-12],
        [ 1.6012e+01, -1.1616e+02,  9.5132e+01],
        [ 3.4120e+01, -2.7805e+02,  2.7496e+02],
        [ 4.3154e+01, -4.0260e+02,  4.5439e+02],
        [-2.5026e+01,  7.7048e+01, -2.6625e+02],
        [-4.0260e+00, -6.0634e+01, -2.1275e+01],
        [ 1.4532e+01, -2.3437e+02,  2.5974e+02],
        [ 4.3798e+01, -2.5647e+02,  2.0086e+02],
        [ 2.2707e+01, -7.7197e+01, -6.1988e+01],
        [-4.6149e+00,  6.6916e+01, -2.8905e+02],
        [ 4.6933e+01, -3.5411e+02,  3.3522e+02]])
    gt = torch.tensor([[   1.0308, -425.3509, -692.8826],
        [ -89.0329, -346.2212, -254.7838],
        [-132.4456,    6.4585,    9.5795],
        [ 132.4459,   -6.4585,   -9.5796],
        [  74.1731, -373.5017, -250.4925],
        [ -15.3735, -497.8137, -678.0806],
        [   0.0000,    0.0000,    0.0000],
        [   2.6789, -133.6875,  191.3945],
        [  -8.7095, -218.8138,  433.7016],
        [  -2.7949, -192.3311,  606.1464],
        [ -86.7290, -297.9068, -133.9241],
        [-122.0166, -249.2365,  110.5208],
        [-151.1074, -207.3004,  384.7039],
        [ 133.8453, -237.7186,  387.5272],
        [ 105.2707, -268.8306,  111.8623],
        [  68.2045, -304.4867, -134.5611],
        [  44.2500, -266.3294,  531.7385]])

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')
    draw_kp_in_3d(
        axis, gt.cpu().numpy(), label='gt',
        marker='o', color='blue'
    )
    draw_kp_in_3d(
        axis, pred.cpu().numpy(), label='pred',
        marker='^', color='red'
    )
    # axis.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    debug()
    1/0
    args = parse_args()
    config = get_config('experiments/human36m/train/human36m_alg.yaml')

    try:
        main(config, args.milestone, args.exp)
    except ZeroDivisionError:
        print('Have you forgotten a breakpoint?')
