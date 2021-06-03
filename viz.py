from pathlib import Path
import torch
import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # https://stackoverflow.com/a/56222305

from post.plots import get_figa
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


def viz_2ds():
    keypoints_2d = torch.tensor([[[ -2.0071, -16.0643],
         [ -1.8683,  -7.9817],
         [ -3.3633,   0.2151],
         [  3.3854,  -0.2165],
         [  2.8818,  -7.9629],
         [  3.3296, -16.1577],
         [  0.0000,   0.0000],
         [  1.2350,   4.2863],
         [  2.4509,   6.7607],
         [  1.8804,   5.5442],
         [ -5.8511,   7.0405],
         [ -3.7357,  10.5431],
         [ -0.5401,   7.9224],
         [  5.6607,   6.5787],
         [  9.5146,  10.4012],
         [ 10.9426,   8.5828],
         [  1.5299,   4.5383]],

        [[ -1.3339, -10.6763],
         [ -1.2740,  -5.4427],
         [ -2.2446,   0.1436],
         [  2.2544,  -0.1442],
         [  1.9651,  -5.4300],
         [  2.2139, -10.7432],
         [  0.0000,   0.0000],
         [  0.8322,   2.8881],
         [  1.6791,   4.6318],
         [  1.3002,   3.8335],
         [ -3.8806,   4.6694],
         [ -2.5137,   7.0943],
         [ -0.3697,   5.4231],
         [  3.8755,   4.5040],
         [  6.4275,   7.0264],
         [  7.2572,   5.6922],
         [  1.0503,   3.1156]],

        [[ -2.6845, -21.4860],
         [ -2.4367, -10.4097],
         [ -4.4795,   0.2865],
         [  4.5188,  -0.2890],
         [  3.7584, -10.3850],
         [  4.4513, -21.6011],
         [  0.0000,   0.0000],
         [  1.6295,   5.6551],
         [  3.1822,   8.7780],
         [  2.4205,   7.1365],
         [ -7.8422,   9.4363],
         [ -4.9353,  13.9286],
         [ -0.7018,  10.2946],
         [  7.3546,   8.5474],
         [ 12.5216,  13.6885],
         [ 14.6665,  11.5037],
         [  1.9826,   5.8811]],

        [[ -1.6027, -12.8274],
         [ -1.5150,  -6.4721],
         [ -2.6924,   0.1722],
         [  2.7065,  -0.1731],
         [  2.3368,  -6.4569],
         [  2.6595, -12.9056],
         [  0.0000,   0.0000],
         [  0.9943,   3.4509],
         [  1.9929,   5.4973],
         [  1.5374,   4.5328],
         [ -4.6664,   5.6149],
         [ -3.0052,   8.4815],
         [ -0.4389,   6.4387],
         [  4.6010,   5.3471],
         [  7.6721,   8.3871],
         [  8.7268,   6.8449],
         [  1.2455,   3.6947]]])

    _, axis = get_figa(1, 1, heigth=10, width=5)
    colors = list(mcolors.TABLEAU_COLORS.values())

    for view_i, color in zip(range(keypoints_2d.shape[0]), colors):
        kps = keypoints_2d[view_i]
        norm = torch.norm(kps, p='fro') * 1e2

        label = 'view #{:0d} norm={:.2f}'.format(view_i, norm)
        draw_kps_in_2d(axis, kps.cpu().numpy(), label=label, color=color)

    axis.set_ylim(axis.get_ylim()[::-1])  # invert
    # axis.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()


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
        [ 2.0776e+02, -5.9902e+02, -1.2716e+02],
        [ 1.0107e+02, -2.7732e+02, -6.4132e+00],
        [-3.9188e+01,  3.5047e+01,  1.1023e+02],
        [ 3.8942e+01, -3.3260e+01, -1.0916e+02],
        [ 1.8480e+02, -3.2464e+02, -1.4518e+02],
        [ 2.8240e+02, -6.3811e+02, -2.4170e+02],
        [-6.8212e-12,  4.5475e-12,  9.3223e-12],
        [ 8.2116e+01,  4.9525e+01,  3.9710e+01],
        [ 2.0538e+02,  5.8142e+01,  5.2160e+01],
        [ 2.1387e+02,  1.1968e+02,  1.0400e+02],
        [ 4.0965e+02, -3.9075e+01,  1.5882e+02],
        [ 2.6055e+02, -1.8248e+01,  2.3033e+02],
        [ 1.4791e+02,  7.9160e+01,  1.5630e+02],
        [ 2.2099e+02,  2.7082e+01, -6.1581e+01],
        [ 2.7039e+02, -1.3718e+02, -2.0417e+02],
        [ 2.1288e+02, -2.1851e+02, -1.3580e+02],
        [ 2.6278e+02,  4.4972e+01,  8.1907e+01]
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
        [ -31.6247, -100.3399, -871.4619],
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
        [  57.6037, -565.6554,  167.7551]
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
    # debug_live_training()
    # debug_noisy_kps()
    # viz_experiment_samples()
    viz_2ds()
