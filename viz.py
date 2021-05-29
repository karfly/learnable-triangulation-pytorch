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
    pred = torch.tensor([[-2.2083e+02, -6.5139e+01, -1.4584e+02],
        [-2.5222e+01, -3.3071e+01, -3.2159e+01],
        [-2.1775e+00,  2.4571e+00, -6.6057e+00],
        [-1.8993e+00, -4.8983e+00,  1.9944e+00],
        [-3.9214e+01, -3.5972e+01, -1.6088e+01],
        [-2.3805e+02, -5.2441e+01, -9.8511e+01],
        [-1.5632e-13,  1.1369e-13,  1.2790e-13],
        [-1.7426e+01, -1.3648e+00, -2.4001e+01],
        [-3.9399e+01, -1.5618e+01, -7.0510e+01],
        [-4.5750e+01, -3.7012e+01, -1.1805e+02],
        [-1.2012e+01, -7.9855e+00, -3.6254e+01],
        [-1.5900e+01,  2.0743e+00, -3.1978e+01],
        [-3.7738e+01, -8.2187e+00, -6.6559e+01],
        [-3.6682e+01, -1.6854e+01, -5.6796e+01],
        [-1.4259e+01, -1.0444e+01, -1.4101e+01],
        [-7.5132e+00, -9.0889e+00, -2.3265e+01],
        [-3.9001e+01, -2.4205e+01, -8.6298e+01]])
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
    draw_kp_in_3d(axis, gt.cpu().numpy(), 'GT (resampled)', 'o', 'blue')
    draw_kp_in_3d(axis, pred.cpu().numpy(), 'prediction', '^', 'red')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    config = get_config('experiments/human36m/train/human36m_alg.yaml')

    try:
        debug()
        # main(config, args.milestone, args.exp)
    except ZeroDivisionError:
        print('Have you forgotten a breakpoint?')
