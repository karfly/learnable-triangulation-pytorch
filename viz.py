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
        (6, 3),  # pelvis -> left anca
        (3, 4),  # left anca -> left knee
        (4, 5),  # left knee -> left foot

        (6, 2),  # pelvis -> right anca
        (2, 1),  # right anca -> right knee
        (1, 0),  # right knee -> right foot

        (6, 7),  # pelvis -> back
        (7, 8),  # back -> neck
        (8, 9),  # neck -> head
        (9, 16),  # head -> nose

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
    pred = torch.tensor([[-7.6816e+01, -1.0128e+02,  2.0695e+01],
        [-4.3430e+01, -4.2793e+01,  3.9446e+00],
        [-4.9607e+00, -3.6847e-01, -6.3809e+00],
        [ 9.5493e-01, -2.6682e+00,  8.1226e+00],
        [-3.6333e+01, -3.3956e+01,  1.8492e+01],
        [-6.8758e+01, -9.1360e+01,  3.1245e+01],
        [ 1.1813e+00,  3.8762e-02,  3.1309e-01],
        [ 1.8361e+01,  1.3239e+01,  3.9560e+00],
        [ 3.7046e+01,  2.8087e+01,  2.6287e+01],
        [ 4.3868e+01,  4.6385e+01,  5.0022e+01],
        [ 1.8518e+01,  3.8230e+01,  4.6319e+01],
        [-1.3389e+00,  9.5479e+00,  1.8709e+01],
        [ 2.8213e+01,  2.1733e+01,  1.7224e+01],
        [ 3.3770e+01,  2.3431e+01,  2.5136e+01],
        [ 2.7387e+01,  3.0527e+01,  3.0452e+01],
        [ 3.4501e+01,  4.6283e+01,  4.2892e+01],
        [ 3.6357e+01,  3.4379e+01,  3.2808e+01]]).float() * 30
    gt = torch.tensor([[  37.4991, -144.7940, -871.4387],
        [ 129.3202,   54.0021, -465.2055],
        [ 138.2558,  -32.2440,   13.5774],
        [-138.2538,   32.2435,  -13.5772],
        [-140.5757,  297.0284, -421.7780],
        [-135.5292,   27.7523, -796.5333],
        [   0.0000,    0.0000,    0.0000],
        [ -45.3304,  -75.7998,  246.8953],
        [ -82.1804,  -68.0094,  504.1621],
        [ -44.4716,  -11.2745,  663.8755],
        [ 146.7015,  168.1520,  560.7039],
        [ 312.8317,   29.3780,  420.4814],
        [  55.7141, -117.3896,  474.8474],
        [-201.8953,  -12.5610,  434.1146],
        [-274.3504,  275.3397,  384.4215],
        [ -98.4383,  254.2147,  571.8475],
        [ -54.6237,   34.4836,  558.8606]]).float()

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    draw_kp_in_3d(
        axis, pred.detach().cpu().numpy(), label='pred',
        marker='^', color='red'
    )

    draw_kp_in_3d(
        axis, gt.detach().cpu().numpy(), label='gt',
        marker='o', color='blue'
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
