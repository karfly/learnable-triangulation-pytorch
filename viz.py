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
    pred = torch.tensor([[-7.1042e+02, -2.7234e+02, -6.7056e+02],
        [-4.1145e+02, -1.6168e+02, -3.2207e+02],
        [-6.1555e+01, -5.5982e+01,  5.5945e+01],
        [ 6.9755e+01,  5.9565e+01, -5.3922e+01],
        [-3.3980e+02, -8.4072e+01, -3.9047e+02],
        [-6.5707e+02, -2.2126e+02, -7.2733e+02],
        [-4.5475e-13,  2.2737e-13,  2.2737e-12],
        [ 2.2121e+01,  9.3021e+01,  1.0262e+02],
        [ 1.5545e+01,  2.0094e+02,  1.5031e+02],
        [ 5.7666e+01,  2.4406e+02,  2.5227e+02],
        [-3.4902e+02,  1.5639e+02,  4.4881e+01],
        [-3.9380e+02,  1.5856e+01,  6.4618e+01],
        [-8.8302e+01,  9.7801e+01,  1.7954e+02],
        [ 1.2056e+02,  2.6594e+02,  9.3584e+01],
        [ 2.6060e+01,  2.1081e+02, -1.5178e+02],
        [-2.0058e+02,  2.7324e+01, -2.5674e+02],
        [-5.3402e+01,  2.1110e+02,  1.4592e+02]]).float()
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
        [  57.6037, -565.6554,  167.7551]]).float()

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    from mvn.pipeline.cam2cam import rotate2gt
    #pred = rotate2gt(pred.unsqueeze(0), gt.unsqueeze(0))[0]
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
