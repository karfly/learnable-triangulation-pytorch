from pathlib import Path
import torch
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # https://stackoverflow.com/a/56222305

from mvn.mini import get_config
from mvn.pipeline.setup import setup_dataloaders
from mvn.utils.tred import create_plane, find_plane_minimizing_normal, rotate_points, find_line_minimizing_normal, create_line, mirror_points_along_x, mirror_points_along_z, mirror_points_along_y, rotx, roty, rotz
from mvn.utils.img import rotation_matrix_from_vectors_rodrigues
from mvn.utils.multiview import euclidean_to_homogeneous, homogeneous_to_euclidean


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
        ['pelvis']
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
            axis.scatter(
                [ xs[point_i] ], [ ys[point_i] ],
                marker='o',
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
    cam_pred = torch.tensor([[
        [-9.7820e-01, -8.5482e-02,  1.8925e-01,  0.0000e+00],
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
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]).float()
    pred = torch.tensor([
        [-3.5775e+02, -6.3525e+02,  1.4934e+03],
        [ 4.5278e-01, -4.3093e+02,  8.1449e+02],
        [-6.3064e+01, -2.4749e+02, -7.8843e+01],
        [ 6.4177e+01,  2.4451e+02,  7.8651e+01],
        [-1.1322e+02, -5.6299e+01,  9.1810e+02],
        [-3.7023e+02, -2.6666e+02,  1.6866e+03],
        [ 5.4570e-12, -1.8190e-12,  9.0949e-13],
        [-2.1489e+01,  6.9028e+01, -5.3477e+02],
        [ 6.7954e+01,  2.9132e+02, -1.1051e+03],
        [ 1.0728e+02,  3.7937e+02, -1.6084e+03],
        [-4.2451e+00, -5.5528e+02, -4.5706e+01],
        [-6.2352e+01, -3.3761e+02, -5.1284e+02],
        [-7.3252e+00, -1.0266e+01, -1.0587e+03],
        [ 1.1019e+02,  4.6226e+02, -7.7651e+02],
        [ 1.5593e+02,  4.2291e+02, -1.1092e+02],
        [ 3.0009e+02,  2.8915e+02,  3.3600e+02],
        [ 1.4575e+02,  3.0650e+02, -1.3460e+03]
    ]).float()
    
    cam_gt = torch.tensor([[
        [-9.2829e-01,  3.7185e-01,  6.5016e-04,  4.9728e-14],
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
        [-170.1615,   32.1489, -845.8221],
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
        [ -21.2917,  -26.4346,  570.1847]
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
    # print('gt cam', R.from_matrix(
    #     cam[:3, :3]
    # ).as_euler('zyx', degrees=True))

    # cam = cam_pred[cam_i]
    # print('pred cam', R.from_matrix(
    #     cam[:3, :3]
    # ).as_euler('zyx', degrees=True))
    # cam[:3, :3] = torch.mm(
    #     rotx(torch.tensor(np.pi)).float(),
    #     cam_pred[cam_i, :3, :3]
    # )
    # print('rotated cam', R.from_matrix(
    #     cam[:3, :3]
    # ).as_euler('zyx', degrees=True))
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
    # print(R.from_matrix(
    #     cam[:3, :3]
    # ).as_euler('zyx', degrees=True))

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
