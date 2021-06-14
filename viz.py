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
from mvn.utils.tred import get_cam_location_in_world, apply_umeyama
from mvn.pipeline.cam2cam import PELVIS_I
from mvn.models.loss import KeypointsMSESmoothLoss


def viz_geodesic():
    """ really appreciate https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html """

    def _gen_some_eulers():
        return np.float32([])

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

    colors = plt.get_cmap('jet')(np.linspace(0, 1, rots.shape[0]))
    for aa, dist, color in zip(
        angle_axis.numpy(),
        distances.numpy(),
        colors):

        label = 'rotate by {:.1f} along [{:.1f}, {:.1f}, {:.1f}] => distance {:.1f}'.format(
            np.degrees(aa[-1]), aa[0], aa[1], aa[2], dist
        )
        axis.plot(
            [0, aa[0]],  # from origin ...
            [0, aa[1]],
            [0, aa[2]],  # ... to vec
            label=label,
            color=color,
        )

    # show axis
    axis.quiver(
        0, 0, 0,
        1, 0, 0,
        normalize=True,
        color='black',
    )
    axis.quiver(
        0, 0, 0,
        0, 1, 0,
        normalize=True,
        color='black',
    )
    axis.quiver(
        0, 0, 0,
        0, 0, 1,
        normalize=True,
        color='black',
    )

    axis.set_xlim3d(-2.0, 2.0)
    axis.set_ylim3d(-2.0, 2.0)
    axis.set_zlim3d(-2.0, 2.0)
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
    xs = np.linspace(0, 1e3, n_points)

    threshold = 1e2

    _, axis = get_figa(1, 1, heigth=12, width=30)

    for alpha in np.linspace(0.1, 0.3, 2):
        for beta in np.linspace(0.9, 1.5, 3):
            ys = smooth(threshold, alpha, beta)(xs.copy())

            axis.plot(
                xs, ys,
                label='smoothed (alpha={:.1f}, beta={:.1f}'.format(alpha, beta)
            )

    axis.plot(xs, xs, label='MSE')

    axis.vlines(x=threshold, ymin=0, ymax=np.max(
        xs), linestyle=':', label='threshold')

    axis.set_xlim((xs[0], xs[-1]))
    axis.set_yscale('log')

    axis.legend(loc='upper left')


def viz_extrinsics():
    convention = 'ZXY'
    eulers = torch.tensor([
        np.deg2rad(30), np.deg2rad(50), 0  # no pitch
    ])
    cam_orient_in_world = euler_angles_to_matrix(eulers, convention)
    aa_in_world = R.from_matrix(cam_orient_in_world.numpy()).as_rotvec()
    print('GT orient', np.rad2deg(eulers.numpy()))

    cam_R = cam_orient_in_world.T  # ext predicted
    aa_in_pose = R.from_matrix(cam_R).as_rotvec()
    print('eulers of predicted pose', np.rad2deg(
        matrix_to_euler_angles(cam_R, convention)
    ))

    print('eulers of predicted orientation', np.rad2deg(
        matrix_to_euler_angles(cam_R.T, convention)
    ))

    fig = plt.figure(figsize=plt.figaspect(1.5))
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    plot_vector(axis, aa_in_world, from_origin=True, color='blue')
    plot_vector(axis, aa_in_pose, from_origin=True, color='red')

    plot_vector(axis, [1, 0, 0])  # X
    plot_vector(axis, [0, 1, 0])  # Y
    plot_vector(axis, [0, 0, 1])  # Z
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
            else:
                marker, s = 'o', 10

            axis.scatter(
                [ xs[point_i] ], [ ys[point_i] ], [ zs[point_i] ],
                marker=marker,
                s=s,
                color=colors[point_i],
                label=label + ' {:.0f}'.format(point_i)
            )

        print(label, 'centroid ~', keypoints_3d.mean(axis=0))
        print(label, 'pelvis ~', keypoints_3d[get_joints_index('pelvis')])


def compare_in_world(try2align=True, force_pelvis_in_origin=True, show_metrics=True):
    def _f(axis, gt, pred):
        if try2align:
            pred = apply_umeyama(
                gt.unsqueeze(0),
                pred.unsqueeze(0),
                scaling=False
            )[0]

        if force_pelvis_in_origin:
            pred = pred - pred[PELVIS_I].unsqueeze(0).repeat(17, 1)

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

        compare_in_world()(
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
        [[ 9.9921e-01,  3.9749e-02,  0.0000e+00,  0.0000e+00],
         [ 9.6489e-03, -2.4255e-01,  9.7009e-01,  0.0000e+00],
         [ 3.8560e-02, -9.6932e-01, -2.4275e-01,  5.1307e+03]],

        [[-4.3500e-01, -9.0043e-01,  0.0000e+00,  0.0000e+00],
         [ 3.4412e-01, -1.6624e-01, -9.2409e-01,  0.0000e+00],
         [ 8.3208e-01, -4.0198e-01,  3.8217e-01,  4.1502e+03]],

        [[-6.0735e-01,  7.9444e-01,  0.0000e+00,  0.0000e+00],
         [ 1.1142e-01,  8.5177e-02,  9.9012e-01,  0.0000e+00],
         [ 7.8659e-01,  6.0134e-01, -1.4025e-01,  4.9509e+03]],

        [[ 2.6125e-01,  9.6527e-01,  0.0000e+00,  0.0000e+00],
         [ 5.4255e-01, -1.4684e-01, -8.2709e-01,  0.0000e+00],
         [-7.9836e-01,  2.1608e-01, -5.6207e-01,  5.8863e+03]],

        [[-2.6902e-01, -9.6313e-01,  0.0000e+00,  0.0000e+00],
         [-1.1010e-01,  3.0753e-02,  9.9344e-01,  0.0000e+00],
         [-9.5682e-01,  2.6726e-01, -1.1431e-01,  5.4901e+03]],

        [[ 7.8607e-01, -6.1814e-01,  0.0000e+00,  0.0000e+00],
         [ 6.1423e-01,  7.8110e-01, -1.1229e-01,  0.0000e+00],
         [ 6.9414e-02,  8.8271e-02,  9.9367e-01,  5.4349e+03]],

        [[-2.0934e-02,  9.9978e-01,  0.0000e+00,  0.0000e+00],
         [ 3.0759e-01,  6.4405e-03, -9.5150e-01,  0.0000e+00],
         [-9.5129e-01, -1.9919e-02, -3.0766e-01,  4.2587e+03]],

        [[-1.0306e-01,  9.9467e-01,  0.0000e+00,  0.0000e+00],
         [ 6.2204e-01,  6.4453e-02,  7.8033e-01,  0.0000e+00],
         [ 7.7617e-01,  8.0423e-02, -6.2537e-01,  4.0056e+03]],

        [[ 8.4393e-01, -5.3645e-01,  0.0000e+00,  0.0000e+00],
         [-4.5701e-02, -7.1897e-02,  9.9636e-01,  0.0000e+00],
         [-5.3450e-01, -8.4087e-01, -8.5193e-02,  5.8545e+03]],

        [[ 5.4398e-01, -8.3910e-01,  0.0000e+00,  0.0000e+00],
         [-7.0250e-01, -4.5542e-01,  5.4688e-01,  0.0000e+00],
         [-4.5889e-01, -2.9749e-01, -8.3721e-01,  5.8835e+03]],

        [[ 5.8735e-01,  8.0933e-01,  0.0000e+00,  0.0000e+00],
         [ 4.4343e-01, -3.2180e-01, -8.3655e-01,  0.0000e+00],
         [-6.7705e-01,  4.9135e-01, -5.4789e-01,  5.7496e+03]],

        [[-7.2616e-01,  6.8753e-01,  0.0000e+00,  0.0000e+00],
         [ 2.8994e-01,  3.0623e-01, -9.0673e-01,  0.0000e+00],
         [-6.2340e-01, -6.5843e-01, -4.2171e-01,  5.9297e+03]],

        [[ 9.8748e-01, -1.5774e-01,  0.0000e+00,  0.0000e+00],
         [ 4.5448e-02,  2.8451e-01, -9.5759e-01,  0.0000e+00],
         [ 1.5105e-01,  9.4561e-01,  2.8812e-01,  5.3503e+03]],

        [[-4.6567e-01, -8.8496e-01,  0.0000e+00,  0.0000e+00],
         [-7.6658e-01,  4.0338e-01,  4.9964e-01,  0.0000e+00],
         [-4.4216e-01,  2.3267e-01, -8.6623e-01,  4.9765e+03]],

        [[-2.1499e-02, -9.9977e-01,  0.0000e+00,  0.0000e+00],
         [-2.9016e-01,  6.2395e-03, -9.5696e-01,  0.0000e+00],
         [ 9.5674e-01, -2.0573e-02, -2.9023e-01,  5.6903e+03]],

        [[ 6.3490e-01, -7.7259e-01,  0.0000e+00,  0.0000e+00],
         [ 5.3252e-01,  4.3761e-01, -7.2451e-01,  0.0000e+00],
         [ 5.5975e-01,  4.5999e-01,  6.8926e-01,  5.7360e+03]],

        [[ 9.0845e-01,  4.1800e-01,  0.0000e+00,  0.0000e+00],
         [-4.0339e-01,  8.7671e-01, -2.6203e-01,  0.0000e+00],
         [-1.0953e-01,  2.3804e-01,  9.6506e-01,  4.0510e+03]],

        [[-4.1878e-01, -9.0809e-01,  0.0000e+00,  0.0000e+00],
         [ 6.8969e-01, -3.1806e-01,  6.5052e-01,  0.0000e+00],
         [-5.9073e-01,  2.7242e-01,  7.5949e-01,  5.8337e+03]],

        [[ 3.3336e-01,  9.4280e-01,  0.0000e+00,  0.0000e+00],
         [ 2.8658e-01, -1.0133e-01,  9.5268e-01,  0.0000e+00],
         [ 8.9819e-01, -3.1758e-01, -3.0397e-01,  5.1374e+03]],

        [[ 1.4927e-01,  9.8880e-01,  0.0000e+00,  0.0000e+00],
         [ 3.4731e-01, -5.2432e-02,  9.3628e-01,  0.0000e+00],
         [ 9.2579e-01, -1.3976e-01, -3.5125e-01,  4.6593e+03]],

        [[ 9.9268e-01, -1.2078e-01,  0.0000e+00,  0.0000e+00],
         [ 2.7105e-02,  2.2278e-01, -9.7449e-01,  0.0000e+00],
         [ 1.1770e-01,  9.6736e-01,  2.2442e-01,  4.0116e+03]],

        [[ 9.9770e-01, -6.7718e-02,  0.0000e+00,  0.0000e+00],
         [-4.9726e-02, -7.3262e-01, -6.7882e-01,  0.0000e+00],
         [ 4.5968e-02,  6.7726e-01, -7.3431e-01,  4.9600e+03]],

        [[ 2.0452e-03,  1.0000e+00,  0.0000e+00,  0.0000e+00],
         [-9.9923e-01,  2.0437e-03, -3.9124e-02,  0.0000e+00],
         [-3.9124e-02,  8.0019e-05,  9.9923e-01,  5.3220e+03]],

        [[-3.4704e-01, -9.3785e-01,  0.0000e+00,  0.0000e+00],
         [-3.4963e-01,  1.2938e-01,  9.2791e-01,  0.0000e+00],
         [-8.7024e-01,  3.2203e-01, -3.7280e-01,  4.7785e+03]],

        [[-6.6245e-01, -7.4911e-01,  0.0000e+00,  0.0000e+00],
         [-1.0630e-01,  9.4001e-02, -9.8988e-01,  0.0000e+00],
         [ 7.4153e-01, -6.5575e-01, -1.4190e-01,  4.4692e+03]],

        [[-8.1011e-02,  9.9671e-01,  0.0000e+00,  0.0000e+00],
         [-9.3217e-01, -7.5765e-02,  3.5401e-01,  0.0000e+00],
         [ 3.5285e-01,  2.8679e-02,  9.3524e-01,  5.6779e+03]],

        [[-8.7514e-01,  4.8386e-01,  0.0000e+00,  0.0000e+00],
         [ 3.7045e-02,  6.7001e-02, -9.9706e-01,  0.0000e+00],
         [-4.8244e-01, -8.7258e-01, -7.6560e-02,  5.2864e+03]],

        [[-8.4756e-01, -5.3070e-01,  0.0000e+00,  0.0000e+00],
         [ 5.1994e-01, -8.3038e-01, -2.0034e-01,  0.0000e+00],
         [ 1.0632e-01, -1.6980e-01,  9.7973e-01,  5.5468e+03]],

        [[ 4.2352e-01,  9.0589e-01,  0.0000e+00,  0.0000e+00],
         [-3.8813e-02,  1.8146e-02, -9.9908e-01,  0.0000e+00],
         [-9.0506e-01,  4.2313e-01,  4.2845e-02,  4.3079e+03]],

        [[ 3.4914e-01,  9.3707e-01,  0.0000e+00,  0.0000e+00],
         [-4.7606e-01,  1.7738e-01,  8.6134e-01,  0.0000e+00],
         [ 8.0713e-01, -3.0073e-01,  5.0803e-01,  4.9743e+03]],

        [[-8.4321e-02, -9.9644e-01,  0.0000e+00,  0.0000e+00],
         [ 8.5679e-01, -7.2504e-02,  5.1054e-01,  0.0000e+00],
         [-5.0873e-01,  4.3050e-02,  8.5985e-01,  4.0966e+03]],

        [[-1.9584e-01,  9.8064e-01,  0.0000e+00,  0.0000e+00],
         [ 8.2190e-01,  1.6414e-01,  5.4547e-01,  0.0000e+00],
         [ 5.3491e-01,  1.0682e-01, -8.3813e-01,  4.9029e+03]],

        [[ 9.9811e-01, -6.1416e-02,  0.0000e+00,  0.0000e+00],
         [-4.6511e-02, -7.5589e-01,  6.5305e-01,  0.0000e+00],
         [-4.0107e-02, -6.5181e-01, -7.5732e-01,  4.9848e+03]],

        [[-9.4563e-01, -3.2525e-01,  0.0000e+00,  0.0000e+00],
         [ 2.5008e-01, -7.2706e-01, -6.3941e-01,  0.0000e+00],
         [ 2.0797e-01, -6.0464e-01,  7.6887e-01,  4.2661e+03]],

        [[-4.6126e-01, -8.8727e-01,  0.0000e+00,  0.0000e+00],
         [ 3.2118e-01, -1.6697e-01,  9.3218e-01,  0.0000e+00],
         [-8.2709e-01,  4.2998e-01,  3.6199e-01,  5.7250e+03]],

        [[ 9.4542e-01, -3.2585e-01,  0.0000e+00,  0.0000e+00],
         [-2.4255e-01, -7.0374e-01,  6.6777e-01,  0.0000e+00],
         [-2.1759e-01, -6.3133e-01, -7.4437e-01,  5.7952e+03]],

        [[ 1.8479e-01, -9.8278e-01,  0.0000e+00,  0.0000e+00],
         [ 5.1677e-01,  9.7169e-02,  8.5059e-01,  0.0000e+00],
         [-8.3594e-01, -1.5718e-01,  5.2583e-01,  4.8936e+03]],

        [[ 6.7236e-01,  7.4022e-01,  0.0000e+00,  0.0000e+00],
         [ 6.1233e-01, -5.5619e-01, -5.6187e-01,  0.0000e+00],
         [-4.1591e-01,  3.7778e-01, -8.2722e-01,  5.4163e+03]],

        [[ 9.9987e-01, -1.5909e-02,  0.0000e+00,  0.0000e+00],
         [ 1.5903e-02,  9.9945e-01,  2.9091e-02,  0.0000e+00],
         [-4.6282e-04, -2.9087e-02,  9.9958e-01,  4.2354e+03]],

        [[ 7.0869e-01,  7.0552e-01,  0.0000e+00,  0.0000e+00],
         [ 5.9534e-01, -5.9801e-01,  5.3662e-01,  0.0000e+00],
         [ 3.7860e-01, -3.8029e-01, -8.4383e-01,  5.4190e+03]],

        [[ 9.9914e-01,  4.1516e-02,  0.0000e+00,  0.0000e+00],
         [-3.5915e-02,  8.6434e-01, -5.0162e-01,  0.0000e+00],
         [-2.0825e-02,  5.0119e-01,  8.6509e-01,  4.7148e+03]],

        [[-8.4446e-01,  5.3562e-01,  0.0000e+00,  0.0000e+00],
         [ 2.2354e-01,  3.5244e-01,  9.0874e-01,  0.0000e+00],
         [ 4.8674e-01,  7.6740e-01, -4.1735e-01,  5.9159e+03]],

        [[ 4.3747e-01,  8.9923e-01,  0.0000e+00,  0.0000e+00],
         [-6.9341e-02,  3.3734e-02,  9.9702e-01,  0.0000e+00],
         [ 8.9655e-01, -4.3617e-01,  7.7111e-02,  4.2041e+03]],

        [[ 6.7988e-02,  9.9769e-01,  0.0000e+00,  0.0000e+00],
         [-9.9766e-01,  6.7986e-02, -7.0160e-03,  0.0000e+00],
         [-6.9998e-03,  4.7700e-04,  9.9998e-01,  5.5148e+03]],

        [[ 5.0129e-02,  9.9874e-01,  0.0000e+00,  0.0000e+00],
         [-9.7602e-01,  4.8989e-02, -2.1211e-01,  0.0000e+00],
         [-2.1184e-01,  1.0633e-02,  9.7725e-01,  5.5544e+03]],

        [[-2.5423e-01, -9.6714e-01,  0.0000e+00,  0.0000e+00],
         [-6.2624e-01,  1.6462e-01,  7.6205e-01,  0.0000e+00],
         [-7.3701e-01,  1.9373e-01, -6.4752e-01,  4.9510e+03]],

        [[ 9.9018e-01,  1.3980e-01,  0.0000e+00,  0.0000e+00],
         [-9.3795e-02,  6.6436e-01, -7.4151e-01,  0.0000e+00],
         [-1.0366e-01,  7.3423e-01,  6.7094e-01,  4.3718e+03]],

        [[-2.6543e-01, -9.6413e-01,  0.0000e+00,  0.0000e+00],
         [-1.7270e-01,  4.7544e-02, -9.8383e-01,  0.0000e+00],
         [ 9.4854e-01, -2.6113e-01, -1.7912e-01,  5.7166e+03]],

        [[-6.0285e-01, -7.9786e-01,  0.0000e+00,  0.0000e+00],
         [ 7.8965e-01, -5.9665e-01,  1.4308e-01,  0.0000e+00],
         [-1.1416e-01,  8.6258e-02,  9.8971e-01,  5.9547e+03]],

        [[ 1.8255e-02,  9.9983e-01,  0.0000e+00,  0.0000e+00],
         [-8.5733e-02,  1.5653e-03,  9.9632e-01,  0.0000e+00],
         [ 9.9615e-01, -1.8188e-02,  8.5747e-02,  4.7480e+03]]
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

    pred = torch.tensor([
        [-5.5953e+02, -1.1585e+02,  1.3756e+03],
        [-1.8078e+02, -1.4087e+02,  7.8004e+02],
        [ 1.4075e+02, -1.3193e+02,  1.5206e+02],
        [-1.4147e+02,  1.3305e+02, -1.5169e+02],
        [-5.4797e+02,  2.0389e+02,  4.9155e+02],
        [-1.0315e+03,  3.8463e+02,  9.4849e+02],
        [ 2.0048e-16,  5.9855e-17, -9.0716e-16],
        [ 2.1958e+02, -4.9226e+01, -3.1443e+02],
        [ 4.5148e+02, -1.0020e+02, -6.7415e+02],
        [ 6.0535e+02, -1.3853e+02, -9.0311e+02],
        [ 3.0231e+02, -1.5303e+02, -1.8716e+02],
        [ 4.2812e+02, -2.8033e+02,  4.2343e+01],
        [ 5.5158e+02, -2.2397e+02, -4.5599e+02],
        [ 2.4481e+02,  4.0157e+01, -6.9642e+02],
        [-9.1513e+01,  1.8718e+02, -4.4682e+02],
        [ 2.0218e+02, -3.1932e+01, -3.7963e+02],
        [ 5.5064e+02, -1.4042e+02, -7.7247e+02]
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

    def _plot_cam_config(axis, gt, pred):
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0, 1, len(pred)))

        locs = get_cam_location_in_world(pred)
        for i, loc in enumerate(locs):
            axis.scatter(
                [ loc[0] ], [ loc[1] ], [ loc[2] ],
                marker='o',
                s=600,
                color=colors[i],
                label='pred cam #{:.0f}'.format(i)
            )
            plot_vector(axis, loc, from_origin=False)

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

    # compare_in_world(
    #     try2align=True,
    #     force_pelvis_in_origin=False,
    #     show_metrics=True
    # )(axis, gt, pred)
    #_compare_in_camspace(cam_i=1)
    #_compare_in_proj(cam_i=0, norm=True)
    _plot_cam_config(axis, None, cam_pred)

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
