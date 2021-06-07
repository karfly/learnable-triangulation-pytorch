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
        [[ 1.8816e-02,  7.7693e-01, -6.2931e-01,  0.0000e+00],
         [ 1.9344e-01, -6.2036e-01, -7.6009e-01,  0.0000e+00],
         [-9.8093e-01, -1.0743e-01, -1.6196e-01,  4.2893e+01]],

        [[-5.9588e-01, -2.6902e-01,  7.5667e-01,  4.5310e+01],
         [-1.7503e-01, -8.7607e-01, -4.4930e-01,  2.5750e+02],
         [ 7.8377e-01, -4.0017e-01,  4.7495e-01, -3.7138e+03]],

        [[ 8.2694e-01,  1.7745e-02, -5.6201e-01,  1.2610e+02],
         [-5.4649e-01,  2.6065e-01, -7.9587e-01, -3.3830e+02],
         [ 1.3237e-01,  9.6527e-01,  2.2524e-01, -3.6862e+03]],

        [[-1.5362e-01,  7.0898e-01, -6.8830e-01, -4.9843e+02],
         [ 5.8830e-01,  6.2528e-01,  5.1276e-01, -1.5283e+02],
         [ 7.9392e-01, -3.2616e-01, -5.1315e-01,  2.6754e+03]]
    ]).float()
    pred = torch.tensor([
        [-3.3592e-02, -1.0706e+01, -1.2980e+02],
        [ 1.8201e+00,  6.3619e+00, -7.1586e+01],
        [ 6.3585e+00,  3.0083e+01, -4.0077e+00],
        [-5.3102e+00, -2.7058e+01,  3.6768e+00],
        [-2.6047e+00, -4.2829e+01, -6.3176e+01],
        [ 8.3736e-01, -5.3501e+01, -1.2423e+02],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 2.1761e-01,  9.2966e+00,  3.8711e+01],
        [ 2.8404e+00,  1.5457e+01,  7.8874e+01],
        [ 1.2772e+01,  4.3608e+01,  1.0353e+02],
        [ 4.6058e+01,  1.6603e+02,  5.4073e+01],
        [ 3.2022e+01,  1.2413e+02,  5.7272e+01],
        [ 9.4980e+00,  4.5044e+01,  6.9667e+01],
        [-5.3885e+00, -1.5309e+01,  6.9801e+01],
        [-1.6102e+01, -5.0383e+01,  3.5808e+01],
        [-1.7162e+01, -6.7019e+01,  3.9130e+00],
        [ 9.4690e+00,  2.9460e+01,  8.6586e+01]
    ]).float()
    
    cam_gt = torch.tensor([
        [[-9.3846e-01,  3.4522e-01,  1.0962e-02, -4.1703e-15],
         [ 8.8055e-02,  2.6982e-01, -9.5888e-01,  4.4515e-14],
         [-3.3398e-01, -8.9890e-01, -2.8361e-01,  5.5126e+03]],

        [[-7.7074e-01,  1.5958e-01, -6.1685e-01,  3.4004e+03],
         [-2.0282e-01,  8.5633e-01,  4.7494e-01, -2.6182e+03],
         [ 6.0401e-01,  4.9116e-01, -6.2764e-01,  9.0696e+03]],

        [[ 7.7327e-01, -1.4673e-01,  6.1686e-01, -3.4005e+03],
         [ 1.3512e-01,  9.8864e-01,  6.5785e-02, -3.6265e+02],
         [-6.1951e-01,  3.2480e-02,  7.8432e-01,  1.4064e+03]],

        [[-9.9450e-01, -9.3654e-02,  4.6829e-02, -2.5815e+02],
         [-4.5798e-02,  7.9123e-01,  6.0980e-01, -3.3616e+03],
         [-9.4162e-02,  6.0430e-01, -7.9117e-01,  8.7841e+03]]
    ]).float()
    gt = torch.tensor([
        [ 1.7703e+01,  1.7006e+02, -9.2785e+02],
        [-6.0124e+01,  9.2687e+01, -4.7959e+02],
        [-1.3127e+02,  5.5741e+01,  3.2727e-01],
        [ 1.3127e+02, -5.5740e+01, -3.2727e-01],
        [ 1.7603e+02, -1.5665e+02, -4.7420e+02],
        [ 2.3222e+02, -1.7232e+02, -9.3200e+02],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
        [-5.3660e+00,  6.2792e+01,  2.5453e+02],
        [ 8.9624e+00,  4.3324e+01,  5.1341e+02],
        [-7.9511e+01, -2.2318e+00,  6.7360e+02],
        [-5.5594e+02, -1.7050e+02,  4.4094e+02],
        [-4.1403e+02,  4.4287e+01,  4.2517e+02],
        [-1.2069e+02,  9.9120e+01,  4.6452e+02],
        [ 1.3914e+02,  2.4199e+01,  4.4269e+02],
        [ 2.9129e+02,  5.7373e+01,  1.8510e+02],
        [ 3.6268e+02, -7.7911e+00, -5.4019e+01],
        [-4.6473e+01, -4.4729e+01,  5.7198e+02]
    ]).float()

    def _compare_in_world():
        draw_kps_in_3d(
            axis, gt.detach().cpu().numpy() * 5, label='gt',
            marker='o', color='blue'
        )
        
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

    _compare_in_world()
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
