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
        [-6.5886e+01, -1.3756e+02, -7.4163e+02],
        [-4.2579e+01, -9.1604e+01, -3.7134e+02],
        [-8.4284e+01, -5.5748e+01, -2.9069e+01],
        [ 8.7061e+01,  5.2137e+01,  2.9758e+01],
        [ 7.3910e+01, -1.2275e+01, -3.4450e+02],
        [ 6.6977e+01, -6.1656e+01, -7.3668e+02],
        [-5.0022e-12,  1.5461e-11, -4.5475e-12],
        [-3.7637e+01,  7.4037e+00,  1.7666e+02],
        [-3.1635e+01,  4.8530e+01,  3.8648e+02],
        [-6.2313e+01,  5.9443e+01,  5.3563e+02],
        [-1.6360e+02, -1.2363e+02, -8.9686e+01],
        [-1.5279e+02, -8.2430e+01,  9.2966e+01],
        [-1.1512e+02, -1.7966e+01,  3.1597e+02],
        [ 5.8077e+01,  8.9793e+01,  3.2978e+02],
        [ 1.2584e+02,  8.9285e+01,  1.3376e+02],
        [ 1.4439e+02,  6.5769e+01, -1.5835e+01],
        [-5.2975e+01,  5.2193e+01,  4.6075e+02]
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
    # viz_2ds()
