from pathlib import Path
import torch
import numpy as np
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # https://stackoverflow.com/a/56222305

from mvn.mini import get_config
from mvn.pipeline.setup import setup_dataloaders


# todo fix legs
def get_joints_connections():
    return [
        (6, 1),  # pelvis -> 
        (1, 0),
        (1, 4),
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


def draw_kp_in_3d(axis, keypoints_3d, label, marker='o', color='blue'):
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
            markersize=10,
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
    pred = torch.tensor([[-7.2219e+02,  9.5125e+02,  1.7432e+02],
        [-2.8474e+02,  5.2597e+02,  1.3553e+02],
        [ 1.5094e+02,  1.1203e+02, -9.6932e+00],
        [-1.4966e+02, -1.1084e+02,  1.4246e+01],
        [-4.6978e+02,  3.8987e+02,  2.4835e+02],
        [-8.7906e+02,  8.4280e+02,  2.6199e+02],
        [ 0.0000e+00,  1.8190e-12,  1.8190e-12],
        [ 1.6403e+02, -5.6716e+01,  4.0144e+02],
        [ 2.6596e+02, -8.0615e+01,  8.8229e+02],
        [ 4.2130e+02, -1.2298e+02,  1.0317e+03],
        [ 3.6629e+02,  2.8448e+02,  1.5357e+03],
        [ 4.0293e+02,  3.9061e+02,  1.1218e+03],
        [ 3.8480e+02,  6.7776e+01,  7.8066e+02],
        [ 9.3483e+01, -2.1252e+02,  8.1100e+02],
        [-2.7844e+02, -1.0885e+02,  6.7267e+02],
        [-3.2237e+02,  1.8415e+02,  4.4313e+02],
        [ 3.1608e+02, -1.1876e+01,  1.0948e+03]])
    gt = torch.tensor([[ -31.6247, -100.3399, -871.4619],
        [ -68.8613,  -88.2665, -418.9454],
        [-130.5779,  -20.4618,   14.3550],
        [ 130.5781,   20.4619,  -14.3550],
        [ 112.0524, -112.4079, -436.4425],
        [ 118.4395, -118.9372, -890.5573],
        [   0.0000,    0.0000,    0.0000],
        [  20.1498, -202.9053,  113.7092],
        [  72.8428, -445.9834,  178.7122],
        [  42.3387, -530.1173,  276.0580],
        [ -16.5688, -832.9115,   36.2845],
        [-165.1364, -633.3756,   -2.1917],
        [ -75.0140, -423.4212,  157.7408],
        [ 206.9062, -377.9351,  164.2977],
        [ 324.1460, -285.8358,  -71.3889],
        [ 168.2460, -202.8033, -250.7503],
        [  57.6037, -565.6554,  167.7551]])

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')
    draw_kp_in_3d(axis, gt.cpu().numpy(), 'GT (resampled)', 'o', 'blue')
    draw_kp_in_3d(axis, pred.cpu().numpy(), 'prediction', '^', 'red')
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
