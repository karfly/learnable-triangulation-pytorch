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
    pred = torch.tensor([[-2.5950e+02,  6.5235e+01, -1.0026e+03],
        [-5.0973e+01,  7.9123e+00, -5.5225e+02],
        [ 1.0012e+02, -6.1195e+01, -7.5836e+01],
        [-9.8671e+01,  6.2012e+01,  7.5625e+01],
        [-3.7767e+02,  1.0200e+02, -3.2387e+02],
        [-7.1215e+02,  1.0734e+02, -6.2512e+02],
        [ 3.9733e-13,  6.0470e-12, -3.1395e-12],
        [ 1.4034e+02, -7.1611e-01,  2.5373e+02],
        [ 3.2685e+02,  5.0220e+01,  4.6809e+02],
        [ 5.4577e+02,  1.3059e+02,  5.5380e+02],
        [ 3.4278e+02,  3.8372e+01,  1.2404e+02],
        [ 3.0873e+02, -9.2713e+01,  2.6452e+01],
        [ 3.7987e+02, -2.2428e+01,  3.5265e+02],
        [ 1.7790e+02,  9.1622e+01,  5.1497e+02],
        [-5.4157e+01,  1.1314e+02,  3.1712e+02],
        [ 2.0619e+02,  1.0312e+02,  2.1795e+02],
        [ 4.3529e+02,  9.7546e+01,  4.7806e+02]])
    gt = torch.tensor([[-223.9388,   56.2951, -865.2037],
        [-197.3877,   30.1725, -428.1547],
        [-108.9469,   97.4755,   10.1214],
        [ 108.9460,  -97.4747,  -10.1213],
        [ 143.1042,  -64.2862, -459.7523],
        [ 263.7129,   67.6030, -860.3243],
        [   0.0000,    0.0000,    0.0000],
        [  13.1453,    7.5346,  260.7762],
        [ -17.3390,  -42.1605,  504.9365],
        [-128.4527, -152.3386,  660.5565],
        [-189.8130,  -77.0542,  259.2988],
        [-214.2607,  158.3594,  199.2538],
        [-111.6482,   91.7718,  462.0903],
        [ 112.6446, -138.1503,  454.3574],
        [ 188.5219, -197.1659,  180.8492],
        [ -46.6052, -188.7412,  246.1689],
        [ -88.4431, -124.6931,  556.3450]])

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
