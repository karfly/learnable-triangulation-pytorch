from pathlib import Path
import torch
import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # https://stackoverflow.com/a/56222305

from scipy.spatial.transform import Rotation as R

from post.plots import get_figa
from mvn.mini import get_config
from mvn.pipeline.setup import setup_dataloaders
from mvn.utils.multiview import build_intrinsics, Camera
from mvn.utils.tred import get_cam_location_in_world, apply_umeyama, rotz, rotation_matrix2axis_angle
from mvn.pipeline.ours import PELVIS_I
from mvn.models.loss import KeypointsMSESmoothLoss, GeodesicLoss


def viz_geodesic():
    """ really appreciate https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html """

    def _gen_some_eulers():
        return np.float64([])

    rots = torch.cat([
        # rotx(torch.tensor(np.pi / 2)).unsqueeze(0),
        # roty(torch.tensor(np.pi / 3)).unsqueeze(0),
        rotz(torch.tensor(np.pi / 2)).unsqueeze(0),
        torch.tensor(R.random().as_matrix()).unsqueeze(0),
        torch.tensor(R.random().as_matrix()).unsqueeze(0),
        torch.tensor(R.random().as_matrix()).unsqueeze(0),
    ])

    distances = GeodesicLoss()._criterion(
        rots.float(),
        torch.eye(3, 3).repeat(rots.shape[0], 1, 1).float().to(rots.device)
    )

    angle_axis = rotation_matrix2axis_angle(rots)

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    for aa, dist, color in zip(
        angle_axis.numpy(),
        distances.numpy(),
        mcolors.TABLEAU_COLORS):

        label = 'rotate by {:.0f}° along [{:.1f}, {:.1f}, {:.1f}]: geodesic distance {:.2f}'.format(
            np.degrees(aa[-1]), aa[0], aa[1], aa[2], dist
        )
        axis.plot(
            [0, aa[0]],  # from origin ...
            [0, aa[1]],
            [0, aa[2]],  # ... to vec
            label=label,
            color=color
        )

    # show axis
    axis.quiver(
        0, 0, 0,
        1, 0, 0,
        normalize=True,
        color='black',
        linestyle='--'
    )
    axis.quiver(
        0, 0, 0,
        0, 1, 0,
        normalize=True,
        color='black',
        linestyle='--'
    )
    axis.quiver(
        0, 0, 0,
        0, 0, 1,
        normalize=True,
        color='black',
        linestyle='--'
    )

    coord_lim = 1.0
    axis.set_xlim3d(-coord_lim, coord_lim)
    axis.set_ylim3d(-coord_lim, coord_lim)
    axis.set_zlim3d(-coord_lim, coord_lim)
    axis.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


def viz_se_smooth():
    def smooth(threshold, alpha, beta):
        def _f(x):
            x[x > threshold] = np.power(
                x[x > threshold],
                alpha
            ) * (threshold ** beta)  # soft version

            return x
        return _f

    n_points = 100
    xs = np.linspace(0, 2e2, n_points)

    threshold = 1e2

    _, axis = get_figa(1, 1, heigth=12, width=30)

    for alpha in np.linspace(0.1, 0.3, 2):
        for beta in np.linspace(0.9, 1.5, 3):
            ys = smooth(threshold, alpha, beta)(xs.copy())
            axis.plot(
                xs, ys,
                label='smoothed (alpha={:.1f}, beta={:.1f})'.format(alpha, beta)
            )

    axis.plot(xs, xs, label='MSE (original)')

    axis.vlines(x=threshold, ymin=0, ymax=np.max(
        xs), linestyle=':', label='threshold')

    axis.set_xlim((xs[0], xs[-1]))
    axis.set_yscale('log')

    axis.legend(loc='upper left')
    axis.set_xlabel('original loss')

    plt.tight_layout()
    plt.show()


def viz_berhu():
    def berhu(c):
        def _f(x):
            out = x.copy()
            out[np.abs(x) <= c] = np.abs(out[np.abs(x) <= c])
            out[np.abs(x) > c] = (np.square(out[np.abs(x) > c]) + np.square(c)) / (2*c)
            return out
        return _f

    xs = np.linspace(-5, 5, 1000)
    _, axis = get_figa(1, 1, heigth=12, width=30)

    for c in np.linspace(0.5, 2.5, 4):
        ys = berhu(c)(xs)
        axis.plot(
            xs, ys,
            label='berHu (threshold={:.3f})'.format(c)
        )

    axis.plot(
        xs, np.square(xs),
        '--',
        label='L2',
    )
    axis.plot(
        xs, np.abs(xs),
        '--',
        label='L1',
    )

    axis.set_xlim((xs[0], xs[-1]))
    axis.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def viz_huber():
    def huber(c):
        def _f(x):
            out = x.copy()
            out[np.abs(x) <= c] = np.square(out[np.abs(x) <= c]) * 0.5
            out[np.abs(x) > c] = c * (np.abs(out[np.abs(x) > c]) - 0.5 * c)
            return out
        return _f

    xs = np.linspace(-5, 5, 1000)
    _, axis = get_figa(1, 1, heigth=12, width=30)

    for c in np.linspace(0.5, 2.5, 4):
        ys = huber(c)(xs)
        axis.plot(
            xs, ys,
            label='berHu (threshold={:.3f})'.format(c)
        )

    axis.plot(
        xs, np.square(xs),
        '--',
        label='L2',
    )
    axis.plot(
        xs, np.abs(xs),
        '--',
        label='L1',
    )

    axis.set_xlim((xs[0], xs[-1]))
    axis.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


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

        (8, 13),  # neck -> shoulder
        (13, 14),  # shoulder -> arm
        (14, 15),  # arm -> hand
        (8, 12),  # neck -> shoulder
        (12, 11),  # shoulder -> arm
        (11, 10)  # arm -> hand
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
            if is_vip(point_i):
                marker, s = 'x', 100
                axis.scatter(
                    [ xs[point_i] ], [ ys[point_i] ],
                    marker=marker,
                    s=s,
                    color=color,
                    label=label + ' {:.0f}'.format(point_i)
                )
            else:
                marker, s = 'o', 10
                axis.scatter(
                    [ xs[point_i] ], [ ys[point_i] ],
                    marker=marker,
                    s=s,
                    color=colors[point_i]
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
                axis.scatter(
                    [ xs[point_i] ], [ ys[point_i] ], [ zs[point_i] ],
                    marker=marker,
                    s=s,
                    color=color,
                    label=label
                )
            else:
                marker, s = 'o', 10
                axis.scatter(
                    [ xs[point_i] ], [ ys[point_i] ], [ zs[point_i] ],
                    marker=marker,
                    s=s,
                    color=colors[point_i]
                )

        print(label, 'centroid ~', keypoints_3d.mean(axis=0))
        print(label, 'pelvis ~', keypoints_3d[get_joints_index('pelvis')])


def compare_in_world(try2align=True, scaling=False, force_pelvis_in_origin=True, show_metrics=True):
    def _f(axis, gt, pred):
        if try2align:
            pred = apply_umeyama(
                gt.unsqueeze(0),
                pred.unsqueeze(0),
                scaling=scaling
            )[0]

        if force_pelvis_in_origin:
            pred = pred - pred[PELVIS_I].unsqueeze(0).repeat(17, 1)
            gt = gt - gt[PELVIS_I].unsqueeze(0).repeat(17, 1)

        draw_kps_in_3d(
            axis, gt.detach().cpu().numpy(), label='gt',
            marker='o', color='blue'
        )

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
    dumps_folder = get_dump_folder(milestone, experiment_name)
    gts, pred, _, dataloader = load_data(config, dumps_folder)

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

        compare_in_world(
            try2align=True,
            scaling=False,
            force_pelvis_in_origin=True,
            show_metrics=True
        )(
            axis,
            torch.FloatTensor(gts[sample_i]),
            torch.FloatTensor(pred[sample_i])
        )
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


def viz_2ds():
    keypoints_2d = torch.tensor([
        [[ 4.8415e-01,  2.1003e+00],
        [-2.7001e-02,  1.1663e+00],
        [-2.8865e-01, -8.5847e-02],
        [ 2.9798e-01,  8.8624e-02],
        [-1.7468e-01,  1.3790e+00],
        [-5.6043e-01,  2.6243e+00],
        [ 0.0000e+00,  0.0000e+00],
        [ 7.6863e-02, -6.8167e-01],
        [ 5.9208e-02, -1.4519e+00],
        [ 2.5522e-02, -1.9191e+00],
        [-4.1919e-01, -5.3182e-01],
        [-5.2262e-01, -5.5556e-01],
        [-2.4472e-01, -1.2919e+00],
        [ 4.3363e-01, -1.3549e+00],
        [ 1.0899e+00, -8.9900e-01],
        [ 9.7792e-01, -1.4139e+00],
        [ 1.3063e-03, -1.6318e+00]],

        [[ 2.3046e+00,  1.0602e+00],
        [ 1.2328e+00,  5.0544e-01],
        [-6.0097e-02,  2.6153e-01],
        [ 6.0097e-02, -2.6153e-01],
        [ 1.3166e+00, -2.2383e-01],
        [ 2.5585e+00, -9.4359e-02],
        [ 0.0000e+00,  0.0000e+00],
        [-6.7088e-01, -5.7899e-04],
        [-1.4296e+00, -7.1418e-02],
        [-1.8864e+00, -1.3718e-01],
        [-5.4844e-01, -3.7660e-01],
        [-5.2444e-01,  3.4245e-01],
        [-1.2682e+00,  1.5023e-01],
        [-1.3374e+00, -2.3499e-01],
        [-9.0852e-01, -5.3400e-01],
        [-1.4061e+00, -1.0684e+00],
        [-1.6061e+00, -3.3483e-01]],
    ])

    _, axis = get_figa(1, 1, heigth=10, width=5)
    colors = list(mcolors.TABLEAU_COLORS.values())

    for view_i, color in zip(range(keypoints_2d.shape[0]), colors):
        kps = keypoints_2d[view_i]
        if view_i == 0:
            kps = (torch.eye(2) * -1).matmul(kps.view(2, 17)).view(17, 2)

        norm = torch.norm(kps, p='fro') * 1e2

        label = 'view #{:0d} norm={:.2f}'.format(view_i, norm)
        draw_kps_in_2d(axis, kps.cpu().numpy(), label=label, color=color)

    #axis.set_ylim(axis.get_ylim()[::-1])  # invert
    axis.legend(loc='lower right')
    
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
        [[-4.2443e-01, -7.4665e-01, -5.1222e-01,  0.0000e+00],
         [ 2.4692e-01, -6.3970e-01,  7.2788e-01,  0.0000e+00],
         [-8.7114e-01,  1.8246e-01,  4.5588e-01,  1.5252e+05]],

        [[ 7.5974e-01, -3.0468e-01,  5.7442e-01, -3.4799e+03],
         [-1.5405e-01,  7.7392e-01,  6.1426e-01, -3.7902e+03],
         [-6.3171e-01, -5.5517e-01,  5.4104e-01,  2.1839e+03]],

        [[ 7.3524e-01, -3.2578e-01,  5.9437e-01, -3.8283e+03],
         [-1.2664e-01,  7.9545e-01,  5.9265e-01, -3.6438e+03],
         [-6.6587e-01, -5.1101e-01,  5.4359e-01,  2.1683e+03]],

        [[ 7.2222e-01, -3.5045e-01,  5.9630e-01, -3.6894e+03],
         [-1.0154e-01,  7.9907e-01,  5.9260e-01, -3.6506e+03],
         [-6.8417e-01, -4.8854e-01,  5.4152e-01,  2.0948e+03]]
    ]).float()
    cam_gt = torch.tensor([
        [[-9.2829e-01,  3.7185e-01,  6.5016e-04,  5.6843e-14],
         [ 1.0662e-01,  2.6784e-01, -9.5755e-01,  0.0000e+00],
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

    pred = torch.tensor([
        [  59.4021, -623.9094,  227.6595],
        [  25.0295, -516.1789,  206.2660],
        [  85.2009, -391.6398,  532.6085],
        [ 118.7767, -309.9534,  269.8331],
        [  32.5972, -571.8669,  249.4450],
        [-186.9228, -535.6912,  258.4842],
        [   0.0000,    0.0000,    0.0000],
        [   4.7671, -694.4681, 1060.4823],
        [ -79.5791, -464.3202,  999.0338],
        [ -69.3573, -510.7137, 1086.2691],
        [-118.9773, -485.7609,  844.2498],
        [  44.1012, -904.0928,  794.9279],
        [ -31.5803, -635.4127,  958.7453],
        [ -93.3054, -592.9530, 1036.1848],
        [ 294.0831, -697.3865,  971.1086],
        [-123.2928, -426.1807,  473.2826],
        [ -70.7583, -533.6038,  997.8352]
    ]).float()
    gt = torch.tensor([
        [ 218.2898, -156.6467, -852.9685],
        [ 195.9178, -127.7015, -416.4887],
        [  10.4303, -135.2365,   -8.0861],
        [ -10.4303,  135.2364,    8.0861],
        [ -79.2696,   96.6730, -433.5346],
        [-294.3813,  103.3477, -815.0237],
        [   0.0000,    0.0000,    0.0000],
        [ -46.1340,  -31.0803,  219.3000],
        [ -15.2406,  -23.8177,  472.7282],
        [ -19.5752,  -34.9721,  629.8514],
        [ 129.8035,  -27.2173,  171.7486],
        [   3.0058, -237.9564,  145.9086],
        [   5.7934, -144.2112,  405.0301],
        [ -57.0594,   95.9318,  414.1426],
        [ -76.4798,  214.5007,  166.1516],
        [ 111.2382,   62.3681,  218.8224],
        [  59.5684,  -14.7534,  548.8994]
    ]).float()

    def _compare_in_camspace(try2align=True, scaling=False, force_pelvis_in_origin=False, from_master=None):
        def _get_Rt(cams, cam_i, from_master):
            if not (from_master is None):
                master = torch.vstack([
                    cams[from_master],
                    torch.tensor([0, 0, 0, 1])
                ])  # ~ 4 x 4, to allow inverse

                full = torch.mm(
                    torch.vstack([
                        cams[cam_i],
                        torch.tensor([0, 0, 0, 1])
                    ]),
                    torch.inverse(master)
                )
                return full[:3, :3], full[:3, 3]
            else:
                return cams[cam_i, :3, :3], cams[cam_i, :3, 3]

        def _f(axis, cam_i, cam_gt, cam_pred, gt, pred):
            R, t = _get_Rt(cam_gt, cam_i, from_master)
            cam = Camera(R, t, K)
            in_cam = cam.world2cam()(gt.detach().cpu())
            if force_pelvis_in_origin:
                in_cam = in_cam - in_cam[PELVIS_I].unsqueeze(0).repeat(17, 1)

            draw_kps_in_3d(
                axis, in_cam.detach().cpu().numpy(), label='gt',
                marker='^', color='blue'
            )

            R, t = _get_Rt(cam_pred, cam_i, from_master)
            cam = Camera(R, t, K)
            other_in_cam = cam.world2cam()(pred.detach().cpu())
            if force_pelvis_in_origin:
                other_in_cam = other_in_cam - other_in_cam[PELVIS_I].unsqueeze(0).repeat(17, 1)
            if try2align:
                other_in_cam = apply_umeyama(
                    in_cam.unsqueeze(0),
                    other_in_cam.unsqueeze(0),
                    scaling=scaling
                )[0]

            draw_kps_in_3d(
                axis, other_in_cam.detach().cpu().numpy(), label='pred',
                marker='^', color='red'
            )

        return _f

    def _compare_in_proj(axis, cam_i, norm=False):
        def _plot(cam, kps, label, color):
            in_proj = cam.world2proj()(kps.detach().cpu())
            if norm:
                in_proj /= torch.norm(in_proj, p='fro')

            draw_kps_in_2d(
                axis, in_proj.cpu().numpy(), label=label, color=color
            )
            print(in_proj[3:9])
            return in_proj  # just for debugging

        cam = Camera(
            cam_gt[cam_i, :3, :3],
            cam_gt[cam_i, :3, 3],
            K
        )
        _plot(cam, gt, 'gt', 'blue')

        cam = Camera(
            cam_pred[cam_i, :3, :3],
            cam_pred[cam_i, :3, 3],
            K
        )
        _plot(cam, pred, 'pred', 'red')

    def _plot_cam_config(axis, gt, pred):
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1, len(pred)))

        locs = get_cam_location_in_world(pred)
        axis.scatter(
            locs[:, 0], locs[:, 1], locs[:, 2],
            marker='o',
            s=600,
        )
        # for i, loc in enumerate(locs):
        #     axis.scatter(
        #         [ loc[0] ], [ loc[1] ], [ loc[2] ],
        #         marker='o',
        #         s=600,
        #         color=colors[i],
        #         label='pred cam #{:.0f}'.format(i)
        #     )
        #     plot_vector(axis, loc, from_origin=False)

        # locs = get_cam_location_in_world(cam_gt)
        # for i, loc in enumerate(locs):
        #     axis.scatter(
        #         [ loc[0] ], [ loc[1] ], [ loc[2] ],
        #         marker='x',
        #         s=600,
        #         color=colors[i],
        #         label='GT cam #{:.0f}'.format(i)
        #     )
        #     plot_vector(axis, loc, from_origin=False)

        plot_vector(axis, [1, 0, 0])  # X
        plot_vector(axis, [0, 1, 0])  # Y
        plot_vector(axis, [0, 0, 1])  # Z

        #axis.legend()

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')
    compare_in_world(
        try2align=False,
        scaling=False,
        force_pelvis_in_origin=True,
        show_metrics=True
    )(axis, gt, pred)

    # _compare_in_camspace(
    #     try2align=True,
    #     scaling=False,
    #     force_pelvis_in_origin=True,
    #     from_master=0
    # )(axis, 1, cam_gt, cam_pred, gt, pred)

    #axis = fig.add_subplot(1, 1, 1)
    #_compare_in_proj(axis, cam_i=0, norm=False)

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
    #debug_noisy_kps()
    #viz_experiment_samples()
    #viz_2ds()
    #viz_geodesic()
    #viz_berhu()
    #viz_huber()
    #viz_se_smooth()
