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
    pred = torch.tensor([[-2.3803e+02, -4.4967e+01, -3.5908e+02],
        [-1.1653e+02,  1.2114e+01, -1.9327e+02],
        [ 3.7219e+01,  6.5244e+01, -4.0665e+01],
        [-3.9091e+01, -6.3966e+01,  4.2090e+01],
        [-2.0659e+02, -9.7366e+01, -7.3921e+01],
        [-2.3399e+02, -1.5346e+02, -2.5434e+02],
        [-1.8190e-12,  9.0949e-13, -2.2169e-12],
        [ 9.2972e+01,  2.3718e+01,  8.7243e+01],
        [ 1.7935e+02,  6.8355e+01,  1.7228e+02],
        [ 2.3859e+02,  9.3511e+01,  2.6336e+02],
        [ 4.4763e+01,  1.2552e+02, -1.2731e+02],
        [ 1.2801e+02,  1.2557e+02, -4.6814e+01],
        [ 1.9176e+02,  1.1981e+02,  8.8997e+01],
        [ 1.2617e+02, -7.1020e+00,  2.1224e+02],
        [ 1.1317e+00, -1.0598e+02,  1.8718e+02],
        [ 5.7444e+01, -1.1592e+01,  1.8678e+02],
        [ 1.9538e+02,  7.9429e+01,  2.2388e+02]])
    gt = torch.tensor([[ -69.7469,   42.8345, -877.1350],
        [-101.9178,  -30.6103, -445.8926],
        [-145.5123,  -16.8526,    3.9381],
        [ 145.5110,   16.8525,   -3.9381],
        [ 190.1256, -139.0983, -425.9868],
        [ 177.8715,  179.2513, -727.4873],
        [   0.0000,    0.0000,    0.0000],
        [ -13.8068,   34.8594,  258.5103],
        [ -61.1871,   -6.1127,  501.5938],
        [ -65.0831,  -76.9353,  710.4856],
        [-302.0739,  -13.4751,  -99.4407],
        [-277.0164,   58.7330,  132.4661],
        [-201.1459,   16.8625,  409.1190],
        [ 104.6385,   16.6094,  476.0288],
        [ 292.2368,   34.8917,  255.7571],
        [ 114.6705, -108.0950,  343.2067],
        [ -53.8591,  -80.4584,  596.0886]])

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
