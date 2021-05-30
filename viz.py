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
    pred = torch.tensor([[ 2.0891e+02, -4.4671e+02,  1.8723e+02],
        [ 1.2173e+02, -2.6208e+02,  5.8154e+01],
        [ 1.7315e+01, -3.3953e+01,  2.7048e+01],
        [-1.2431e+01,  3.2481e+01, -2.5503e+01],
        [ 7.5922e+01, -2.0420e+02,  5.1061e+01],
        [ 1.7154e+02, -4.2579e+02,  1.5801e+02],
        [ 4.5475e-13, -4.5475e-13, -1.0232e-12],
        [-3.7746e+01,  1.1389e+02, -6.0182e+00],
        [-7.1526e+01,  2.5251e+02, -3.4453e+01],
        [-8.2729e+01,  3.4533e+02, -4.1903e+01],
        [ 5.6911e+01, -1.0896e+02,  3.5106e+01],
        [ 1.6555e+00,  2.6071e+01,  2.5432e+01],
        [-4.9306e+01,  1.8410e+02, -6.3286e+00],
        [-7.0855e+01,  2.3311e+02, -5.0689e+01],
        [-3.1108e+01,  1.0388e+02, -5.4150e+01],
        [ 2.8353e+01, -1.6452e+01, -6.2873e+01],
        [-7.4281e+01,  2.9121e+02, -5.0868e+01]])
    gt = torch.tensor([[-170.1615,   32.1489, -845.8221],
        [ -96.7913, -140.3269, -432.0922],
        [-131.5567,    9.4015,  -16.7268],
        [ 131.5569,   -9.4015,   16.7268],
        [  80.0030,    6.4952, -422.8695],
        [  40.1287,   65.7422, -871.4266],
        [   0.0000,    0.0000,    0.0000],
        [ -40.6688,   61.3262,  221.5786],
        [  -5.4884,   47.9403,  475.8856],
        [ -28.3539,   36.5607,  666.1389],
        [-259.0923, -110.9618,  -75.9137],
        [-223.7264,   10.1488,  141.9137],
        [-138.6906,   48.7752,  404.7029],
        [ 117.5744,   37.9555,  388.8961],
        [ 198.7848,  -38.3537,  133.2452],
        [ 221.5172, -202.4289,  -56.3129],
        [ -21.2917,  -26.4346,  570.1847]])

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
