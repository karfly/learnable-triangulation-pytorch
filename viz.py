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
        (4, 5),
        (6, 3),  # pelvis -> left anca
        (3, 4),  # left anca -> left knee
        (6, 7),  # pelvis -> back
        (6, 2),  # pelvis -> right anca
        (2, 1),  # right anca -> right knee
        (1, 0),  # right knee -> right foot
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


def create_plane(coefficients):
    X, Y = np.meshgrid(
        np.arange(-1e2, 1e2),
        np.arange(-1e2, 1e2)
    )
    Z = C[0] * X + C[1] * Y + C[2]
    # matrix version: Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
    return X, Y, Z


def debug():
    # pred = torch.tensor([[-1.0706e+03, -4.8821e+02, -4.8714e+02],
    #     [-6.7551e+02, -1.5577e+02, -2.8707e+01],
    #     [-6.7588e+01,  1.9132e+02, -3.2873e+01],
    #     [ 6.8114e+01, -1.9156e+02,  3.0584e+01],
    #     [-4.9589e+02, -5.2600e+02,  3.5124e+02],
    #     [-9.0278e+02, -6.9919e+02, -2.0832e+02],
    #     [-3.4106e-13,  0.0000e+00,  0.0000e+00],
    #     [ 3.5770e+02,  1.2433e+02, -6.5918e+01],
    #     [ 7.0596e+02,  2.4474e+02,  1.8443e+01],
    #     [ 8.7034e+02,  3.8278e+02,  1.7224e+02],
    #     [ 5.6701e+02,  5.0892e+02,  4.9911e+02],
    #     [ 3.1217e+02,  6.6246e+02,  2.2334e+02],
    #     [ 5.8807e+02,  4.1310e+02, -6.0947e+01],
    #     [ 6.8485e+02,  3.2514e+01,  7.8479e+01],
    #     [ 6.0543e+02, -1.6426e+02,  5.5381e+02],
    #     [ 7.2707e+02,  1.8527e+02,  6.1494e+02],
    #     [ 7.3246e+02,  2.8876e+02,  2.2314e+02]])
    # pred_a = torch.tensor(
    #     [[ 2.8584e+02,  1.0671e+03, -6.3363e+02],
    #     [ 2.4322e+02,  6.4355e+02, -8.9955e+01],
    #     [ 1.9927e+02, -4.6474e+01,  1.9569e+01],
    #     [-1.9925e+02,  4.5759e+01, -2.1819e+01],
    #     [-2.3798e+02,  7.4690e+02,  1.7735e+02],
    #     [-3.7270e+01,  1.0816e+03, -4.1961e+02],
    #     [ 1.8362e-13,  2.8725e-13, -9.6869e-15],
    #     [-7.6970e+01, -3.7614e+02, -1.8665e+01],
    #     [-1.8497e+02, -7.1641e+02,  1.0564e+02],
    #     [-1.9483e+02, -8.9885e+02,  2.9629e+02],
    #     [-4.8605e-01, -6.5146e+02,  6.3657e+02],
    #     [ 3.2207e+02, -5.6280e+02,  4.0707e+02],
    #     [ 3.2893e+01, -7.1681e+02,  7.2737e+01],
    #     [-3.5945e+02, -5.7987e+02,  1.0384e+02],
    #     [-5.8094e+02, -3.3051e+02,  5.0350e+02],
    #     [-3.7540e+02, -6.0096e+02,  6.6259e+02],
    #     [-2.0826e+02, -7.2592e+02,  3.1518e+02]]
    # )
    # ph = torch.tensor([[ 4.2446e+01, -2.8405e+02, -1.2407e+03],
    #     [ 1.8193e+02,  7.1857e+01, -6.6569e+02],
    #     [ 1.9726e+02, -5.3055e+01,  2.2950e+01],
    #     [-1.9790e+02,  5.0782e+01, -2.2990e+01],
    #     [-1.9838e+02,  4.7970e+02, -6.1358e+02],
    #     [-2.0052e+02,  2.1171e+00, -1.1433e+03],
    #     [ 1.6193e-13,  3.8820e-14, -2.9765e-13],
    #     [-6.5959e+01, -1.2396e+02,  3.5782e+02],
    #     [-1.1919e+02, -9.8638e+01,  7.3122e+02],
    #     [-6.4268e+01,  1.3196e+01,  9.6404e+02],
    #     [ 2.1574e+02,  3.5005e+02,  8.1273e+02],
    #     [ 4.4990e+02,  8.5324e+01,  6.1359e+02],
    #     [ 7.8205e+01, -1.8825e+02,  6.9184e+02],
    #     [-2.9055e+02, -5.6963e+00,  6.2592e+02],
    #     [-3.8840e+02,  4.9953e+02,  5.4758e+02],
    #     [-1.3499e+02,  4.9375e+02,  8.2408e+02],
    #     [-7.7272e+01,  9.2873e+01,  8.0937e+02]])
    # gt = torch.tensor([[  37.4991, -144.7940, -871.4387],
    #     [ 129.3202,   54.0021, -465.2055],
    #     [ 138.2558,  -32.2440,   13.5774],
    #     [-138.2538,   32.2435,  -13.5772],
    #     [-140.5757,  297.0284, -421.7780],
    #     [-135.5292,   27.7523, -796.5333],
    #     [   0.0000,    0.0000,    0.0000],
    #     [ -45.3304,  -75.7998,  246.8953],
    #     [ -82.1804,  -68.0094,  504.1621],
    #     [ -44.4716,  -11.2745,  663.8755],
    #     [ 146.7015,  168.1520,  560.7039],
    #     [ 312.8317,   29.3780,  420.4814],
    #     [  55.7141, -117.3896,  474.8474],
    #     [-201.8953,  -12.5610,  434.1146],
    #     [-274.3504,  275.3397,  384.4215],
    #     [ -98.4383,  254.2147,  571.8475],
    #     [ -54.6237,   34.4836,  558.8606]])

    pred = torch.tensor([[-5.8390e+02,  4.4225e+02,  2.6058e+02],
        [-3.6810e+02,  1.8690e+02,  1.3985e+02],
        [-1.6339e+01, -1.2069e+01,  3.4503e+01],
        [ 1.6738e+01,  1.1775e+01, -3.4833e+01],
        [-3.3782e+02,  1.0517e+02,  2.0229e+01],
        [-5.8530e+02,  2.1482e+02,  2.8833e+01],
        [ 1.1369e-12, -1.7053e-13, -9.0949e-13],
        [ 2.1887e+02, -6.6050e+01, -3.8939e+01],
        [ 4.8875e+02, -1.5829e+02, -8.7451e+01],
        [ 6.6111e+02, -2.1744e+02, -1.1437e+02],
        [ 1.8605e+02, -1.4040e+02, -8.0904e+01],
        [ 1.1178e+02, -8.0142e+01,  1.9492e+01],
        [ 3.9562e+02, -1.4896e+02, -4.9342e+01],
        [ 4.9281e+02, -1.2883e+02, -1.0737e+02],
        [ 4.1133e+02, -4.9429e+01, -1.3143e+02],
        [ 6.0237e+02, -1.7731e+02, -2.0835e+02],
        [ 5.6795e+02, -2.0929e+02, -1.2395e+02]])
    gt = torch.tensor([[-230.7124, -112.5173, -824.0206],
        [ -25.2134,  -83.7622, -438.2794],
        [  54.1768, -124.6038,    1.3616],
        [ -54.1767,  124.6037,   -1.3616],
        [ 109.1408,  124.0950, -419.1916],
        [ 236.8049,  105.9139, -837.7878],
        [   0.0000,    0.0000,    0.0000],
        [ -38.3125,  -29.7123,  220.9904],
        [ -41.7408,  -56.5128,  474.9648],
        [ -34.1479,  -68.7751,  629.2502],
        [ 155.8149,   24.6664,  202.7643],
        [ 105.3489, -211.0617,  147.6121],
        [  26.1948, -157.2762,  406.0334],
        [-129.6425,   50.6015,  457.0524],
        [-270.4420,  255.6727,  338.4919],
        [-174.6004,  369.3506,  536.0985],
        [   3.6715,    5.2748,  549.8020]])

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    draw_kp_in_3d(
        axis, pred.cpu().numpy(), label='pred',
        marker='^', color='red'
    )
    draw_kp_in_3d(
        axis, gt.cpu().numpy(), label='gt',
        marker='o', color='blue'
    )
    # draw_kp_in_3d(
    #     axis, pred.cpu().numpy(), label='pred',
    #     marker='^', color='red'
    # )
    # draw_kp_in_3d(
    #     axis, pred_a.cpu().numpy(), label='pred a',
    #     marker='^', color='orange'
    # )
    # draw_kp_in_3d(
    #     axis, ph.cpu().numpy() / 1.4, label='pred a',
    #     marker='^', color='red'
    # )

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
