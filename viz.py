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
from mvn.utils.multiview import build_intrinsics, Camera
from mvn.utils.tred import get_cam_location_in_world, rotate_points, get_centroid, apply_umeyama
from mvn.pipeline.cam2cam import PELVIS_I
from mvn.models.loss import KeypointsMSESmoothLoss


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
    K = build_intrinsics(
        translation=(0, 0),
        f=(1e2, 1e2),
        shear=0
    )

    cam_pred = torch.tensor([
        [[ 1.5227e-01,  5.9660e-02,  9.8654e-01,  0.0000e+00],
         [ 7.7107e-01, -6.3161e-01, -8.0818e-02,  0.0000e+00],
         [ 6.1828e-01,  7.7299e-01, -1.4218e-01, -4.6307e+03]],

        [[-4.1619e-01, -5.9598e-01, -6.8673e-01,  0.0000e+00],
         [ 9.0908e-01, -2.5688e-01, -3.2801e-01,  0.0000e+00],
         [ 1.9080e-02, -7.6080e-01,  6.4871e-01, -4.6938e+03]],

        [[-4.2669e-01, -5.4653e-01, -7.2058e-01,  0.0000e+00],
         [-8.0858e-01,  5.8745e-01,  3.3248e-02,  0.0000e+00],
         [ 4.0513e-01,  5.9684e-01, -6.9257e-01,  4.8035e+03]],

        [[-2.2679e-01, -1.2974e-02, -9.7386e-01,  0.0000e+00],
         [ 9.6618e-01, -1.2899e-01, -2.2328e-01,  0.0000e+00],
         [-1.2272e-01, -9.9156e-01,  4.1788e-02, -3.7292e+03]]
    ]).float()
    pred = torch.tensor([
        [ 2.5268e+00,  1.0658e+02, -1.5335e+02],
        [ 2.1525e+01,  6.2747e+01, -7.4802e+01],
        [ 3.0199e+01, -9.3599e+00, -9.7415e+00],
        [-3.0106e+01,  9.5660e+00,  9.3942e+00],
        [-2.4891e+01,  8.6474e+01, -5.1346e+01],
        [-4.2937e+01,  1.3697e+02, -1.2766e+02],
        [-7.0193e-15,  1.0485e-15,  5.4697e-15],
        [-1.6657e+00, -4.2573e+01,  2.7528e+01],
        [-8.7589e+00, -8.6447e+01,  6.5320e+01],
        [-3.7841e+00, -1.1567e+02,  8.6816e+01],
        [ 4.7347e+01, -9.3373e+01,  1.0550e+02],
        [ 6.4868e+01, -5.0244e+01,  7.1611e+01],
        [ 1.9747e+01, -8.8779e+01,  5.2620e+01],
        [-3.4625e+01, -6.9464e+01,  5.8752e+01],
        [-5.6127e+01, -2.3745e+01,  1.9699e+01],
        [-6.4762e+01,  2.7373e+01, -4.0060e+00],
        [ 1.0279e+01, -9.7189e+01,  7.4316e+01]
    ]).float()
    
    cam_gt = torch.tensor([
        [[-9.2829e-01,  3.7185e-01,  6.5016e-04,  4.9728e-14],
         [ 1.0662e-01,  2.6784e-01, -9.5755e-01, -2.7233e-14],
         [-3.5624e-01, -8.8881e-01, -2.8828e-01,  5.5426e+03]],

        [[ 9.3246e-01,  3.6046e-01,  2.4059e-02,  1.9896e-14],
         [ 1.2453e-01, -2.5819e-01, -9.5803e-01, -3.6986e-14],
         [-3.3912e-01,  8.9633e-01, -2.8564e-01,  5.7120e+03]],

        [[-9.5123e-01, -3.0488e-01, -4.7087e-02, -2.3485e-14],
         [-3.5426e-02,  2.5958e-01, -9.6507e-01, -6.5800e-15],
         [ 3.0645e-01, -9.1633e-01, -2.5772e-01,  5.6838e+03]],

        [[ 9.2061e-01, -3.7942e-01,  9.2279e-02,  2.9982e-15],
         [-5.2180e-02, -3.5374e-01, -9.3389e-01,  9.9662e-15],
         [ 3.8698e-01,  8.5493e-01, -3.4546e-01,  4.4827e+03]]
    ]).float()
    gt = torch.tensor([
        [-123.5816,  -63.1810, -864.4102],
        [  23.1837,  -74.3185, -451.8706],
        [  32.6581, -131.7078,   -7.0428],
        [ -32.6581,  131.7076,    7.0428],
        [  19.8084,  157.5186, -437.7446],
        [ -98.6799,  185.7531, -858.4759],
        [   0.0000,    0.0000,    0.0000],
        [ -40.0030,  -28.9358,  220.7925],
        [ -71.0292,  -23.5397,  474.2511],
        [ -74.5952,  -62.3802,  627.2699],
        [ 161.1007, -191.2156,  611.1978],
        [ 228.8489, -214.3185,  374.4852],
        [ -32.2064, -152.6614,  437.6392],
        [-115.1191,   90.9162,  407.3412],
        [-158.4828,  178.7477,  149.7747],
        [-114.1517,  251.6376,  -82.3426],
        [ -23.7438, -100.7664,  531.5288]
    ]).float()

    def _compare_in_world(try2align=True, show_metrics=True):
        fig = plt.figure(figsize=plt.figaspect(1.5))
        axis = fig.add_subplot(1, 1, 1, projection='3d')

        def _f(gt, pred):
            draw_kps_in_3d(
                axis, gt.detach().cpu().numpy(), label='gt',
                marker='o', color='blue'
            )

            if try2align:
                pred = apply_umeyama(gt.unsqueeze(0), pred.unsqueeze(0))[0]
                pred - pred[PELVIS_I].unsqueeze(0).repeat(17, 1)

            draw_kps_in_3d(
                axis, pred.detach().cpu().numpy(), label='pred',
                marker='^', color='red'
            )

            if show_metrics:
                criterion = KeypointsMSESmoothLoss(threshold=20*20)
                loss = criterion(pred.unsqueeze(0), gt.unsqueeze(0))
                print(
                    'loss ({}) = {:.3f}'.format(
                        str(criterion), loss
                    )
                )

                per_pose_error_relative = torch.sqrt(
                    ((gt - pred) ** 2).sum(1)
                ).mean(0)
                print(
                    'MPJPE (relative 2 pelvis) = {:.3f} mm'.format(
                        per_pose_error_relative
                    )
                )

        return _f

    def _compare_in_camspace(cam_i):
        fig = plt.figure(figsize=plt.figaspect(1.5))
        axis = fig.add_subplot(1, 1, 1, projection='3d')

        cam = Camera(
            cam_gt[cam_i, :3, :3],
            cam_gt[cam_i, :3, 3],
            K
        )
        in_cam = cam.world2cam()(gt.detach().cpu())
        draw_kps_in_3d(
            axis, in_cam.detach().cpu().numpy(), label='gt',
            marker='^', color='blue'
        )

        cam = Camera(
            cam_pred[cam_i, :3, :3],
            cam_pred[cam_i, :3, 3],
            K
        )
        in_cam = cam.world2cam()(pred.detach().cpu())
        draw_kps_in_3d(
            axis, in_cam.detach().cpu().numpy(), label='pred',
            marker='^', color='red'
        )

    def _compare_in_proj(cam_i, norm=True):
        _, axis = get_figa(1, 1, heigth=10, width=5)
        K = build_intrinsics(
            translation=(0, 0),
            f=(1e2, 1e2),
            shear=0
        )

        cam = Camera(
            cam_gt[cam_i, :3, :3],
            cam_gt[cam_i, :3, 3],
            K
        )
        in_proj = cam.world2proj()(gt.detach().cpu())
        if norm:
            in_proj /= torch.pow(torch.norm(in_proj, p='fro'), 0.1)

        draw_kps_in_2d(axis, in_proj.cpu().numpy(), label='gt', color='blue')

        cam = Camera(
            cam_pred[cam_i, :3, :3],
            cam_pred[cam_i, :3, 3],
            K
        )
        in_proj = cam.world2proj()(pred.detach().cpu())
        if norm:
            in_proj /= torch.pow(torch.norm(in_proj, p='fro'), 0.1)

        draw_kps_in_2d(axis, in_proj.cpu().numpy(), label='pred', color='red')

        axis.set_ylim(axis.get_ylim()[::-1])  # invert
        #axis.legend(loc='lower right')

    def _plot_cam_config():
        fig = plt.figure(figsize=plt.figaspect(1.5))
        axis = fig.add_subplot(1, 1, 1, projection='3d')
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

    _compare_in_world()(gt, pred)
    #_compare_in_camspace(cam_i=1)
    #_compare_in_proj(cam_i=0, norm=True)
    #_plot_cam_config()

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
