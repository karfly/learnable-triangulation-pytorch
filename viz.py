from pathlib import Path
import torch
import numpy as np
import argparse

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


def get_joints_index(joint_name):
    indices = {
        'pelvis': 6,
        'head': 9,

        'left anca': 3,
        'left knee': 4,
        'left foot': 5,

        'right anca': 2,
        'right knee': 1,
        'right foot': 0,
    }

    return indices[joint_name]


def is_vip(joint_i):
    vips = map(
        get_joints_index,
        ['pelvis', 'head']
    )

    return joint_i in vips


def draw_kps_in_2d(axis, keypoints_2d, label, marker='o', color='blue'):
    for _, joint_pair in enumerate(get_joints_connections()):
        joints = [
            keypoints_2d[joint_pair[0]],
            keypoints_2d[joint_pair[1]]
        ]
        xs = joints[0][0], joints[1][0]
        ys = joints[0][1], joints[1][1]

        axis.plot(
            xs, ys,
            marker=marker,
            markersize=0 if label else 10,
            color=color,
        )

    if label:
        xs = keypoints_2d[:, 0]
        ys = keypoints_2d[:, 1]
        n_points = keypoints_2d.shape[0]

        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1, n_points))
        for point_i in range(n_points):
            if is_vip(point_i):
                marker, s = 'x', 100
            else:
                marker, s = 'o', 10
            axis.scatter(
                [ xs[point_i] ], [ ys[point_i] ],
                marker=marker,
                s=s,
                color=colors[point_i],
                label=label + ' {:.0f}'.format(point_i)
                # todo too many label=label,
            )


def draw_kps_in_3d(axis, keypoints_3d, label=None, marker='o', color='blue'):
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
            if is_vip(point_i):
                marker, s = 'x', 100
            else:
                marker, s = 'o', 10

            axis.scatter(
                [ xs[point_i] ], [ ys[point_i] ], [ zs[point_i] ],
                marker=marker,
                s=s,
                color=colors[point_i],
                label=label + ' {:.0f}'.format(point_i)
                # todo too many label=label,
            )

        print(label, 'centroid ~', keypoints_3d.mean(axis=0))
        print(label, 'pelvis ~', keypoints_3d[get_joints_index('pelvis')])


def viz_experiment_samples():
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

    args = parse_args()
    milestone, experiment_name = args.milestone, args.exp
    config = get_config('experiments/human36m/train/human36m_alg.yaml')

    try:
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
    except ZeroDivisionError:
        print('Have you forgotten a breakpoint?')


def debug_live_training():
    cam_pred = torch.tensor([
        [[-1.0426e-01,  1.4902e-01, -9.8332e-01,  0.0000e+00],
         [ 3.3611e-01,  9.3582e-01,  1.0618e-01,  0.0000e+00],
         [ 9.3603e-01, -3.1944e-01, -1.4765e-01,  1.3526e+04],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-3.1036e-01, -8.6959e-03,  9.5058e-01,  0.0000e+00],
         [-2.0233e-01,  9.7765e-01, -5.7115e-02,  0.0000e+00],
         [-9.2884e-01, -2.1006e-01, -3.0518e-01,  6.3418e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-7.8511e-01, -2.4302e-01, -5.6969e-01,  0.0000e+00],
         [-2.6996e-01,  9.6211e-01, -3.8386e-02,  0.0000e+00],
         [ 5.5743e-01,  1.2365e-01, -8.2096e-01,  1.0749e+04],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-9.3258e-01, -2.8283e-01,  2.2427e-01,  0.0000e+00],
         [-3.4451e-01,  8.8284e-01, -3.1922e-01,  0.0000e+00],
         [-1.0771e-01, -3.7496e-01, -9.2076e-01,  3.7537e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]).float()
    pred = torch.tensor([
        [-3.8543e+02, -2.6456e+02,  1.7456e+02],
        [-4.7330e+02, -2.6061e+02,  1.6107e+02],
        [-2.6539e+02, -1.8158e+02,  3.8059e+01],
        [ 2.5902e+02,  1.9342e+02, -3.7980e+01],
        [-6.7208e+02, -4.3380e+02,  2.8106e+02],
        [-4.0620e+02, -3.0146e+02,  2.3270e+02],
        [ 5.1854e-15,  1.7814e-14, -7.0911e-15],
        [ 7.9106e+02,  6.1649e+02, -2.4878e+02],
        [ 1.4618e+04,  1.1064e+04, -4.9473e+03],
        [-4.1105e+03, -2.9333e+03,  1.4652e+03],
        [ 6.6777e+03,  4.6636e+03, -1.5957e+03],
        [ 1.9031e+03,  1.3543e+03, -6.0402e+02],
        [-4.3704e+03, -3.0855e+03,  1.3382e+03],
        [ 2.6465e+03,  2.1389e+03, -9.0863e+02],
        [ 9.2685e+02,  7.5356e+02, -2.6181e+02],
        [ 6.7747e+02,  5.0830e+02, -1.3347e+02],
        [-3.6536e+03, -2.6322e+03,  1.2236e+03]
    ]).float()
    
    cam_gt = torch.tensor([[
        [-9.3846e-01,  3.4522e-01,  1.0962e-02, -4.1703e-15],
         [ 8.8055e-02,  2.6982e-01, -9.5888e-01,  4.4515e-14],
         [-3.3398e-01, -8.9890e-01, -2.8361e-01,  5.5126e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 9.4337e-01,  3.3147e-01,  1.3482e-02,  1.5904e-14],
         [ 1.0712e-01, -2.6589e-01, -9.5803e-01, -2.9033e-14],
         [-3.1397e-01,  9.0522e-01, -2.8634e-01,  5.6097e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-9.4462e-01, -3.2714e-01, -2.5777e-02,  2.4028e-14],
         [-6.1720e-02,  2.5427e-01, -9.6516e-01, -2.3498e-14],
         [ 3.2230e-01, -9.1013e-01, -2.6038e-01,  5.7300e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 9.0941e-01, -4.1069e-01,  6.5619e-02, -8.0881e-15],
         [-9.1010e-02, -3.5047e-01, -9.3214e-01, -2.5128e-14],
         [ 4.0582e-01,  8.4173e-01, -3.5610e-01,  4.4227e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
    ]).float()
    gt = torch.tensor([
        [ -79.5401, -636.6202,  -37.0452],
        [ -80.1006, -342.1938,  287.2476],
        [-134.9715,    8.6331,   13.0857],
        [ 134.9714,   -8.6331,  -13.0857],
        [ 123.5578, -341.4131,  287.5503],
        [ 132.1298, -641.1887,  -31.6872],
        [   0.0000,    0.0000,    0.0000],
        [  51.0227,  177.0769,  131.2592],
        [ 106.6453,  294.1788,  351.3190],
        [  84.2807,  248.4880,  481.9431],
        [-230.4554,  277.3011,  -61.3511],
        [-153.6896,  433.7529,  114.1089],
        [ -23.4383,  343.8069,  339.6841],
        [ 245.7708,  285.6294,  341.7256],
        [ 396.1953,  433.1156,  164.0866],
        [ 430.9688,  338.0313,  -61.5378],
        [  67.0051,  198.7636,  379.6946]
    ]).float()

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    # compare in world
    draw_kps_in_3d(
        axis, gt.detach().cpu().numpy(), label='gt',
        marker='o', color='blue'
    )
    draw_kps_in_3d(
        axis, pred.detach().cpu().numpy(), label='pred',
        marker='^', color='red'
    )

    # n_cameras = cam_pred.shape[0]
    # for cam_i in range(n_cameras):
    #     eulers = R.from_matrix(
    #         cam_pred[cam_i, :3, :3]
    #     ).as_euler('zyx', degrees=True)
    #     print('pr cam #{:.0f} eulers ~ {}'.format(cam_i, str(eulers)))
    # for cam_i in range(n_cameras):
    #     eulers = R.from_matrix(
    #         cam_gt[cam_i, :3, :3]
    #     ).as_euler('zyx', degrees=True)
    #     print('gt cam #{:.0f} eulers ~ {}'.format(cam_i, str(eulers)))

    # draw points in cam spaces
    # n_cameras = cam_pred.shape[0]
    # colors = plt.get_cmap('cool')(np.linspace(0, 1, n_cameras))
    # for cam_i in range(n_cameras):
    #     in_cam = homogeneous_to_euclidean(
    #         euclidean_to_homogeneous(
    #             gt  # [x y z] -> [x y z 1]
    #         ) @ cam_gt[cam_i].T
    #     )

    #     draw_kps_in_3d(
    #         axis, in_cam.detach().cpu().numpy(), label='cam #{:.0f}'.format(cam_i),
    #         marker='^', color=colors[cam_i]
    #     )

    # compare in cam space
    # cam_i = 3

    # cam = cam_gt[cam_i]
    # in_cam = homogeneous_to_euclidean(
    #     euclidean_to_homogeneous(
    #         gt  # [x y z] -> [x y z 1]
    #     ) @ cam.T
    # )
    # draw_kps_in_3d(
    #     axis, in_cam.detach().cpu().numpy(), label='gt',
    #     marker='^', color='blue'
    # )

    # cam = cam_pred[cam_i]
    # cam[:3, :3] = torch.mm(
    #     rotx(torch.tensor(np.pi)).float(),
    #     cam_pred[cam_i, :3, :3]
    # )
    # in_cam = homogeneous_to_euclidean(
    #     euclidean_to_homogeneous(
    #         pred  # [x y z] -> [x y z 1]
    #     ) @ cam.T
    # )
    # in_cam = mirror_points_along_y(
    #     np.mean([cam_gt[cam_i, 2, 2], cam_pred[cam_i, 2, 2]])
    # )(in_cam)
    # draw_kps_in_3d(
    #     axis, in_cam.detach().cpu().numpy(), label='pred',
    #     marker='^', color='red'
    # )

    # fit, errors, residual = find_plane_minimizing_normal(pred)
    # X, Y, Z = create_plane(fit)
    # axis.scatter(X, Y, Z, color='gray')

    # axis.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


def debug_noisy_kps():
    pred = torch.tensor([[-2.4766e+00,  1.3749e+02],
        [ 5.1553e+00,  6.4850e+01],
        [ 2.0758e+01, -5.5261e+00],
        [-2.1199e+01,  5.6435e+00],
        [-2.6096e+01,  6.9830e+01],
        [-2.7770e+01,  1.4269e+02],
        [-7.5650e-16,  8.0752e-15],
        [-1.5507e+01, -2.8643e+01],
        [-3.7743e+01, -4.8863e+01],
        [-3.7260e+01, -6.8515e+01],
        [-4.3409e+01, -4.1714e+01],
        [-1.0379e+01, -2.9870e+01],
        [-1.2607e+01, -4.6328e+01],
        [-5.6277e+01, -4.2062e+01],
        [-7.1047e+01,  3.4976e+00],
        [-4.0396e+01,  3.5121e+01],
        [-4.1566e+01, -5.1796e+01]])

    gt = torch.tensor([[ -4.2729, 135.4911],
        [  7.2749,  65.5788],
        [ 20.6505,  -8.0638],
        [-22.5586,   5.5275],
        [-30.7718,  69.5852],
        [-28.9555, 139.2640],
        [ -0.5923,  -3.4187],
        [-15.7863, -32.1939],
        [-35.3697, -47.2574],
        [-41.1945, -67.7720],
        [-46.1246, -44.4364],
        [-13.1253, -29.5808],
        [-13.6145, -43.1209],
        [-54.4943, -42.5870],
        [-71.2272,   4.1981],
        [-41.6380,  34.4177],
        [-40.1495, -48.8374]])

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1)

    draw_kps_in_2d(axis, pred.detach().cpu().numpy(), label='gt', marker='^', color='red')
    draw_kps_in_2d(axis, gt.detach().cpu().numpy(), label='gt', marker='o', color='blue')

    axis.set_ylim(axis.get_ylim()[::-1])  # invert
    # axis.legend(loc='lower left')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    debug_live_training()
    # debug_noisy_kps()
    # viz_experiment_samples()
