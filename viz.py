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
from mvn.utils.multiview import homogeneous_to_euclidean, euclidean_to_homogeneous, build_intrinsics
from mvn.utils.tred import get_cam_location_in_world, get_cam_orientation_in_world
from mvn.pipeline.cam2cam import PELVIS_I


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

            draw_kps_in_3d(axis, gts[sample_i], 'GT (resampled)', 'o', 'blue')
            draw_kps_in_3d(axis, pred[sample_i], 'prediction', '^', 'red')
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
    keypoints_2d = torch.tensor([
        [[ 4.2062e+00,  6.7325e+00],
        [ 2.0345e+00, -3.5230e+00],
        [-2.8494e+00, -1.8568e-01],
        [ 2.7873e+00,  1.8163e-01],
        [ 6.5186e+00, -3.7257e+00],
        [ 9.0576e+00,  6.2431e+00],
        [ 6.6884e-17,  2.2233e-16],
        [-1.7581e-01, -4.0769e+00],
        [ 4.0783e-01, -9.4050e+00],
        [ 6.0908e-01, -1.1891e+01],
        [-6.9443e+00, -6.1852e-01],
        [-6.2157e+00, -5.2997e+00],
        [-2.5951e+00, -9.4108e+00],
        [ 3.1765e+00, -9.2050e+00],
        [ 4.3549e+00, -6.6090e+00],
        [ 5.2991e+00, -1.7056e+00],
        [ 4.6859e-01, -9.4208e+00]],

        [[ 4.1949e+00,  6.0977e+00],
        [ 1.7903e+00, -3.1798e+00],
        [-2.7495e+00, -4.9575e-02],
        [ 2.7858e+00,  4.6203e-02],
        [ 5.8071e+00, -3.6465e+00],
        [ 8.2556e+00,  5.7024e+00],
        [ 3.1506e-15,  2.6259e-14],
        [-3.3759e-01, -4.1778e+00],
        [ 4.0149e-01, -9.8858e+00],
        [ 6.8256e-01, -1.2303e+01],
        [-7.5806e+00, -1.3962e-01],
        [-7.1787e+00, -5.0212e+00],
        [-2.8316e+00, -9.5914e+00],
        [ 3.4574e+00, -1.0041e+01],
        [ 5.0321e+00, -7.6827e+00],
        [ 5.8696e+00, -2.1291e+00],
        [ 4.4599e-01, -9.6818e+00]],
    ])

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


# todo refactor
def plot_vector(axis, vec, from_origin=True, color='black'):
    if from_origin:
        axis.quiver(
            0, 0, 0,
            *vec,
            normalize=False,
            length=1e3,
            color=color
        )
    else:
        axis.quiver(
            *vec,
            0, 0, 0,
            normalize=False,
            length=1e3,
            color=color
        )


def debug_live_training():
    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    cam_pred = torch.tensor([
        [[ 9.8445e-01,  1.6897e-01,  4.8062e-02,  0.0000e+00],
         [-1.7014e-01,  9.8519e-01,  2.1336e-02,  0.0000e+00],
         [-4.3745e-02, -2.9182e-02,  9.9862e-01,  3.1667e+03]],

        [[ 6.6179e-01, -7.4882e-01,  3.6176e-02,  4.1043e+03],
         [-7.3553e-01, -6.5787e-01, -1.6189e-01, -1.5858e+03],
         [ 1.4502e-01,  8.0527e-02, -9.8615e-01,  7.7063e+03]],

        [[ 2.6916e-02,  9.5598e-01,  2.9220e-01, -2.1047e+03],
         [ 4.6340e-02,  2.9079e-01, -9.5566e-01,  2.2251e+03],
         [-9.9856e-01,  3.9263e-02, -3.6473e-02,  3.1375e+03]],

        [[-6.1914e-01,  1.6464e-02,  7.8511e-01, -3.3826e+03],
         [-7.8460e-01,  2.8695e-02, -6.1934e-01,  9.3224e+02],
         [-3.2726e-02, -9.9945e-01, -4.8490e-03,  9.2459e+03]]
    ]).float()
    pred = torch.tensor([
        [-4083.6418,  3915.7295,  1557.1318],
        [-3430.3773,  3856.7957,  1663.1131],
        [-3026.4948,  4015.4840,  1920.1767],
        [-2375.4727,  3257.4710,  1343.6151],
        [-2873.3638,  3292.9052,  1256.5397],
        [-3532.7789,  3424.9937,  1218.2702],
        [-2641.6863,  3569.7371,  1580.6659],
        [-2330.0338,  3333.0740,  1437.0582],
        [-2073.0514,  3072.0143,  1265.4970],
        [-2001.3203,  3089.6357,  1304.6766],
        [-2490.6517,  3296.5990,  1397.1004],
        [-2965.8477,  3825.8963,  1761.0065],
        [-2429.9658,  3474.9533,  1553.8458],
        [-1895.8728,  2851.9695,  1102.2948],
        [-2017.1182,  2754.7068,   974.5944],
        [-2475.5139,  3076.2147,  1154.4262],
        [-2100.9863,  3070.9106,  1264.1030]
    ]).float()
    
    cam_gt = torch.tensor([
        [[-9.2829e-01,  3.7185e-01,  6.5016e-04,  4.9728e-14],
         [ 1.0662e-01,  2.6784e-01, -9.5755e-01, -2.7233e-14],
         [-3.5624e-01, -8.8881e-01, -2.8828e-01,  5.5426e+03]],

        [[-7.3154e-01,  1.7293e-01, -6.5950e-01,  3.6554e+03],
         [-2.1223e-01,  8.6149e-01,  4.6130e-01, -2.5568e+03],
         [ 6.4792e-01,  4.7743e-01, -5.9351e-01,  9.0016e+03]],

        [[ 7.6961e-01, -1.3799e-01,  6.2342e-01, -3.4554e+03],
         [ 1.2878e-01,  9.8985e-01,  6.0114e-02, -3.3319e+02],
         [-6.2539e-01,  3.4022e-02,  7.7957e-01,  1.3629e+03]],

        [[-9.9562e-01, -9.1829e-02, -1.7330e-02,  9.6054e+01],
         [-8.3708e-02,  7.9393e-01,  6.0222e-01, -3.3379e+03],
         [-4.1542e-02,  6.0103e-01, -7.9814e-01,  8.9065e+03]]
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

    def _compare_in_world(gt, pred, force_pelvis_in_origin=True):
        draw_kps_in_3d(
            axis, gt.detach().cpu().numpy() * 5, label='gt',
            marker='o', color='blue'
        )
        
        if force_pelvis_in_origin:
            pred = pred - pred[PELVIS_I].unsqueeze(0).repeat(17, 1)

        draw_kps_in_3d(
            axis, pred.detach().cpu().numpy() * 5, label='pred',
            marker='^', color='red'
        )

    def _compare_in_camspace(cam_i):
        K = build_intrinsics(
            translation=(0, 0),
            f=(1e2, 1e2),
            shear=0
        )

        cam = torch.cat([
            cam_gt[cam_i],
            torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(0)
        ], dim=0)
        in_cam = homogeneous_to_euclidean(
            euclidean_to_homogeneous(
                gt  # [x y z] -> [x y z 1]
            ) @ cam.T
        )
        print(in_cam)
        in_proj = torch.mm(in_cam, torch.tensor(K.T).float())  # just apply intrinsic
        print(in_proj)
        print(homogeneous_to_euclidean(in_proj))

        draw_kps_in_3d(
            axis, in_cam.detach().cpu().numpy(), label='gt',
            marker='^', color='blue'
        )

        cam = torch.cat([
            cam_pred[cam_i],
            torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(0)
        ], dim=0)
        in_cam = homogeneous_to_euclidean(
            euclidean_to_homogeneous(
                pred  # [x y z] -> [x y z 1]
            ) @ cam.T
        )
        print(in_cam)
        in_proj = torch.mm(in_cam, torch.tensor(K.T).float())  # just apply intrinsic
        print(in_proj)
        print(homogeneous_to_euclidean(in_proj))
        draw_kps_in_3d(
            axis, in_cam.detach().cpu().numpy(), label='pred',
            marker='^', color='red'
        )

    def _plot_cam_config():
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1, 5))

        locs = get_cam_location_in_world(cam_pred)
        for i, loc in enumerate(locs):
            axis.scatter(
                [ loc[0] ], [ loc[1] ], [ loc[2] ],
                marker='o',
                s=600,
                color=colors[i],
                label='pred cam #{:.0f}'.format(i)
            )
            plot_vector(axis, loc, from_origin=False)

        locs = get_cam_location_in_world(cam_gt)
        for i, loc in enumerate(locs):
            axis.scatter(
                [ loc[0] ], [ loc[1] ], [ loc[2] ],
                marker='x',
                s=600,
                color=colors[i],
                label='GT cam #{:.0f}'.format(i)
            )
            plot_vector(axis, loc, from_origin=False)

        plot_vector(axis, [1, 0, 0])  # X
        plot_vector(axis, [0, 1, 0])  # Y
        plot_vector(axis, [0, 0, 1])  # Z

        axis.legend()

    _compare_in_world(gt, pred)
    # _compare_in_camspace(0)  # the others are master2others ...
    # _plot_cam_config()

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
