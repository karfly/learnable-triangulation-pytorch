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


def debug():
    pred = torch.tensor([[ 4.7978e+02, -4.4164e+02, -3.8788e+02],
        [ 2.9600e+02, -2.0384e+02, -2.3515e+02],
        [ 5.8534e+01,  7.7366e+01, -6.5027e+01],
        [-5.9911e+01, -7.8681e+01,  6.4558e+01],
        [ 2.4255e+02, -3.3462e+02, -6.2346e+01],
        [ 4.8151e+02, -5.6101e+02, -1.8774e+02],
        [-3.6380e-12,  3.4106e-12,  2.8422e-12],
        [-1.8263e+02,  1.4113e+02,  8.3566e+01],
        [-3.6722e+02,  2.8819e+02,  2.1823e+02],
        [-4.1866e+02,  4.5055e+02,  2.8077e+02],
        [ 9.0806e+01,  5.9240e+02,  4.5678e+01],
        [-4.9422e+01,  4.9387e+02,  1.8071e+01],
        [-2.6782e+02,  3.3694e+02,  1.2622e+02],
        [-3.9127e+02,  1.6201e+02,  2.3836e+02],
        [-3.0843e+02, -8.5862e+01,  1.5757e+02],
        [-1.7055e+02, -2.5311e+02,  9.9787e+01],
        [-3.5324e+02,  3.6498e+02,  2.5681e+02]])
    gt = torch.tensor([[ 1.7703e+01,  1.7006e+02, -9.2785e+02],
        [-6.0124e+01,  9.2687e+01, -4.7959e+02],
        [-1.3127e+02,  5.5741e+01,  3.2727e-01],
        [ 1.3127e+02, -5.5740e+01, -3.2727e-01],
        [ 1.7603e+02, -1.5665e+02, -4.7420e+02],
        [ 2.3222e+02, -1.7232e+02, -9.3200e+02],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
        [-5.3660e+00,  6.2792e+01,  2.5453e+02],
        [ 8.9624e+00,  4.3324e+01,  5.1341e+02],
        [-7.9511e+01, -2.2318e+00,  6.7360e+02],
        [-5.5594e+02, -1.7050e+02,  4.4094e+02],
        [-4.1403e+02,  4.4287e+01,  4.2517e+02],
        [-1.2069e+02,  9.9120e+01,  4.6452e+02],
        [ 1.3914e+02,  2.4199e+01,  4.4269e+02],
        [ 2.9129e+02,  5.7373e+01,  1.8510e+02],
        [ 3.6268e+02, -7.7911e+00, -5.4019e+01],
        [-4.6473e+01, -4.4729e+01,  5.7198e+02]])

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
    args = parse_args()
    config = get_config('experiments/human36m/train/human36m_alg.yaml')

    try:
        debug()
        # main(config, args.milestone, args.exp)
    except ZeroDivisionError:
        print('Have you forgotten a breakpoint?')
