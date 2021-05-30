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
        (6, 1),  # pelvis -> right knee
        (1, 0),  # right knee -> right foot
        (6, 4),  # pelvis -> left knee
        (4, 5),
        (6, 3),
        (6, 2),
        (6, 7),
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
    # for joint_pair in get_joints_connections():
    #     joints = [
    #         keypoints_3d[joint_pair[0]],
    #         keypoints_3d[joint_pair[1]]
    #     ]
    #     xs = joints[0][0], joints[1][0]
    #     ys = joints[0][1], joints[1][1]
    #     zs = joints[0][2], joints[1][2]

    xs = keypoints_3d[:, 0]
    ys = keypoints_3d[:, 1]
    zs = keypoints_3d[:, 2]

    if label:
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

    axis.plot(
        xs, ys, zs,
        marker=marker,
        markersize=0 if label else 5,
        color=color,
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


def debug():
    pred = torch.tensor([[ 5.7474e+02, -6.0941e+02, -4.4499e+02],
        [ 2.2752e+02, -2.7802e+02, -1.7834e+02],
        [-6.3767e+01,  1.4216e+00,  2.9491e+01],
        [ 7.8955e+01, -1.5531e+01, -4.8524e+01],
        [ 3.0102e+02, -2.1495e+02, -1.4573e+02],
        [ 5.8171e+02, -5.1567e+02, -3.7614e+02],
        [ 5.1159e-13,  4.5475e-13,  6.8212e-13],
        [-8.3141e+01,  7.8967e+01,  2.5867e+01],
        [-1.3131e+02,  1.1014e+02, -2.5499e+01],
        [-1.5985e+02,  9.0292e+01, -1.1300e+02],
        [-2.9785e+02,  1.8234e+02,  1.2720e+02],
        [-3.2246e+02,  1.5128e+02,  1.7606e+02],
        [-2.1052e+02,  1.2548e+02,  5.1625e+01],
        [-3.3569e+01,  8.6227e+01, -7.3297e+01],
        [ 2.6379e+01,  8.5374e+01, -7.8202e+01],
        [-1.2984e+02,  1.2679e+02, -3.9831e+01],
        [-1.6108e+02,  1.2545e+02, -1.8298e+01]])
    gt = torch.tensor([[  11.5845,  -27.6364,  -66.5652],
        [ -65.4557,   39.7026,  -23.2532],
        [-142.4960,  107.0416,   20.0588],
        [-219.5363,  174.3807,   63.3708],
        [-296.5766,  241.7197,  106.6828],
        [-373.6168,  309.0588,  149.9948],
        [-450.6571,  376.3978,  193.3068],
        [-527.6974,  443.7368,  236.6188],
        [-604.7377,  511.0759,  279.9308],
        [-681.7779,  578.4149,  323.2428],
        [ 629.0713, -567.3667, -413.7159]])

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')
    draw_kp_in_3d(
        axis, gt.cpu().numpy(),
        marker='o', color='blue'
    )
    draw_kp_in_3d(
        axis, pred.cpu().numpy(), label='pred',
        marker='^', color='red'
    )
    axis.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    config = get_config('experiments/human36m/train/human36m_alg.yaml')

    try:
        debug()
        # main(config, args.milestone, args.exp)
    except ZeroDivisionError:
        print('Have you forgotten a breakpoint?')
