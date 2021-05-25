from pathlib import Path
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from mpl_toolkits.mplot3d import Axes3D  # https://stackoverflow.com/a/56222305
from mvn.mini import get_config
from mvn.pipeline.setup import setup_dataloaders


def get_joints_connections():
    return [
        (6, 1),
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


def draw_kp_in_2d(axis, keypoints_2d_in_view, color):
    for joint_pair in get_joints_connections():
        joints = [
            keypoints_2d_in_view[joint_pair[0]],
            keypoints_2d_in_view[joint_pair[1]]
        ]
        xs = joints[0][0], joints[1][0]
        ys = joints[0][1], joints[1][1]
        axis.plot(
            xs, ys,
            marker='o',
            markersize=15,
            color=color,
        )


def draw_kp_in_3d(axis, keypoints_3d, marker='o', color='blue'):
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
            markersize=15,
            color=color,
        )


def load_data(config, dumps_folder=Path('~/_tmp/').expanduser()):
    def _load(file_name):
        f_path = dumps_folder / file_name
        return torch.load(f_path).cpu().numpy()

    keypoints_3d_gt = _load('keypoints_3d_gt.trc')  # see `cam2cam:_save_stuff`
    keypoints_3d_pred = _load('keypoints_3d_pred.trc')
    indices = _load('batch_indexes')  # [0, 3, 2, 4, 1]
    _, val_dataloader, _ = setup_dataloaders(config, distributed_train=False)  # ~ 0 seconds

    return keypoints_3d_gt, keypoints_3d_pred, indices, val_dataloader


def main(config):
    fig = plt.figure()
    axis = fig.gca(projection='3d')

    gts, pred, indices, dataloader = load_data(config)

    scalar_metric, full_metric = dataloader.dataset.evaluate(
        pred,
        indices_predicted=indices,
        split_by_subject=True
    )  # (average 3D MPJPE (relative to pelvis), all MPJPEs)

    print(scalar_metric)  # full_metric

    draw_kp_in_3d(axis, gts[4], 'o', 'blue')  # todo also others samples
    draw_kp_in_3d(axis, pred[4], '^', 'red')

    plt.show()


if __name__ == '__main__':
    config = get_config('experiments/human36m/train/human36m_alg.yaml')

    try:
        main(config)
    except ZeroDivisionError:
        print('Have you forgotten a breakpoint?')
