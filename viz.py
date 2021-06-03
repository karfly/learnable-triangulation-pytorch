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
from mvn.utils.multiview import homogeneous_to_euclidean, euclidean_to_homogeneous


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
    keypoints_2d = torch.tensor([[[-16.4621,  11.0777],
         [ -8.7619,   7.0920],
         [ -2.2775,  -1.5802],
         [  2.1864,   1.5170],
         [ -5.6967,   7.1440],
         [-11.5442,   6.0178],
         [  0.0000,   0.0000],
         [  3.9395,  -0.3826],
         [  8.6280,  -0.4923],
         [ 11.6279,  -2.9150],
         [ -0.0983,   8.9764],
         [ -1.0055,   2.2332],
         [  5.3962,  -1.8452],
         [ 10.4620,   1.5216],
         [  9.6394,   8.4670],
         [  7.4874,   8.4563],
         [  9.8146,  -0.3217]],

        [[-11.1741,   7.5193],
         [ -5.8531,   4.7376],
         [ -1.5079,  -1.0462],
         [  1.4674,   1.0181],
         [ -3.8770,   4.8620],
         [ -8.0410,   4.1916],
         [  0.0000,   0.0000],
         [  2.5885,  -0.2514],
         [  5.5658,  -0.3176],
         [  7.4393,  -1.8650],
         [ -0.0622,   5.6823],
         [ -0.6420,   1.4260],
         [  3.4631,  -1.1842],
         [  6.8076,   0.9901],
         [  6.3142,   5.5462],
         [  4.7998,   5.4210],
         [  6.2673,  -0.2055]],

        [[-21.5646,  14.5114],
         [-11.6589,   9.4369],
         [ -3.0580,  -2.1217],
         [  2.8959,   2.0093],
         [ -7.4436,   9.3347],
         [-14.7592,   7.6938],
         [  0.0000,   0.0000],
         [  5.3305,  -0.5177],
         [ 11.9021,  -0.6792],
         [ 16.1841,  -4.0572],
         [ -0.1384,  12.6403],
         [ -1.4025,   3.1150],
         [  7.4853,  -2.5596],
         [ 14.3005,   2.0799],
         [ 13.0848,  11.4932],
         [ 10.3986,  11.7443],
         [ 13.6885,  -0.4487]],

        [[-13.3122,   8.9581],
         [ -7.0180,   5.6805],
         [ -1.8145,  -1.2589],
         [  1.7562,   1.2185],
         [ -4.6140,   5.7862],
         [ -9.4793,   4.9414],
         [  0.0000,   0.0000],
         [  3.1242,  -0.3034],
         [  6.7666,  -0.3861],
         [  9.0735,  -2.2746],
         [ -0.0762,   6.9592],
         [ -0.7837,   1.7405],
         [  4.2188,  -1.4426],
         [  8.2481,   1.1996],
         [  7.6303,   6.7022],
         [  5.8497,   6.6067],
         [  7.6498,  -0.2508]]])

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
        [[-9.7820e-01, -8.5482e-02,  1.8925e-01,  0.0000e+00],
         [ 1.3602e-01, -9.5238e-01,  2.7288e-01,  0.0000e+00],
         [ 1.5691e-01,  2.9267e-01,  9.4325e-01,  5.6545e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 8.4163e-01,  4.8745e-02,  5.3786e-01,  0.0000e+00],
         [ 2.5049e-01, -9.1755e-01, -3.0880e-01,  0.0000e+00],
         [ 4.7846e-01,  3.9462e-01, -7.8444e-01,  5.7320e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-8.7837e-01, -9.0128e-02, -4.6941e-01,  0.0000e+00],
         [-2.1810e-02, -9.7348e-01,  2.2772e-01,  0.0000e+00],
         [-4.7749e-01,  2.1026e-01,  8.5311e-01,  5.9238e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 9.3676e-01,  1.5488e-01, -3.1384e-01,  0.0000e+00],
         [ 1.1282e-02, -9.0964e-01, -4.1524e-01,  0.0000e+00],
         [-3.4980e-01,  3.8544e-01, -8.5386e-01,  4.5888e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
    ]).float()
    pred = torch.tensor([
        [ 2.3641e+01, -5.1126e+01, -6.3698e+02],
        [-5.3702e+01,  3.0688e+02, -2.9269e+02],
        [-1.4042e+02,  5.0906e+00, -9.4177e+00],
        [ 1.3941e+02, -5.7484e+00,  1.1342e+01],
        [ 1.6083e+02,  3.1911e+02, -2.6591e+02],
        [ 2.4219e+02, -3.2007e+01, -6.1190e+02],
        [ 0.0000e+00, -9.0949e-13, -9.0949e-13],
        [ 1.8934e+01,  1.2938e+02,  1.6864e+02],
        [ 4.9195e+01,  3.6470e+02,  2.6948e+02],
        [ 2.3851e+01,  5.1713e+02,  2.2665e+02],
        [-2.7127e+02, -9.6756e+01,  2.1958e+02],
        [-2.2832e+02,  7.5233e+01,  3.5988e+02],
        [-9.4716e+01,  3.4057e+02,  2.9174e+02],
        [ 1.9795e+02,  3.6118e+02,  2.8243e+02],
        [ 3.4374e+02,  1.5595e+02,  4.3662e+02],
        [ 3.9459e+02, -7.0551e+01,  3.7564e+02],
        [ 1.7937e+01,  4.0214e+02,  1.8694e+02]
    ]).float()
    
    cam_gt = torch.tensor([
        [[-9.2829e-01,  3.7185e-01,  6.5016e-04,  4.9728e-14],
         [ 1.0662e-01,  2.6784e-01, -9.5755e-01, -2.7233e-14],
         [-3.5624e-01, -8.8881e-01, -2.8828e-01,  5.5426e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 9.3246e-01,  3.6046e-01,  2.4059e-02,  1.9896e-14],
         [ 1.2453e-01, -2.5819e-01, -9.5803e-01, -3.6986e-14],
         [-3.3912e-01,  8.9633e-01, -2.8564e-01,  5.7120e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-9.5123e-01, -3.0488e-01, -4.7087e-02, -2.3485e-14],
         [-3.5426e-02,  2.5958e-01, -9.6507e-01, -6.5800e-15],
         [ 3.0645e-01, -9.1633e-01, -2.5772e-01,  5.6838e+03],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 9.2061e-01, -3.7942e-01,  9.2279e-02,  2.9982e-15],
         [-5.2180e-02, -3.5374e-01, -9.3389e-01,  9.9662e-15],
         [ 3.8698e-01,  8.5493e-01, -3.4546e-01,  4.4827e+03],
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
    # draw_kps_in_3d(
    #     axis, gt.detach().cpu().numpy(), label='gt',
    #     marker='o', color='blue'
    # )
    # draw_kps_in_3d(
    #     axis, pred.detach().cpu().numpy(), label='pred',
    #     marker='^', color='red'
    # )

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
    cam_i = 3

    cam = cam_gt[cam_i]
    in_cam = homogeneous_to_euclidean(
        euclidean_to_homogeneous(
            gt  # [x y z] -> [x y z 1]
        ) @ cam.T
    )
    draw_kps_in_3d(
        axis, in_cam.detach().cpu().numpy(), label='gt',
        marker='^', color='blue'
    )

    cam = cam_pred[cam_i]
    in_cam = homogeneous_to_euclidean(
        euclidean_to_homogeneous(
            pred  # [x y z] -> [x y z 1]
        ) @ cam.T
    )
    draw_kps_in_3d(
        axis, in_cam.detach().cpu().numpy(), label='pred',
        marker='^', color='red'
    )

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
    # viz_2ds()
