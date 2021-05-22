from mvn.models.utils import get_grad_params
import torch

import numpy as np

from mvn.pipeline.utils import get_kp_gt, backprop
from mvn.utils.misc import live_debug_log
from mvn.utils.multiview import triangulate_points_in_camspace, euclidean_to_homogeneous
from mvn.models.loss import geo_loss, L2_R_loss, t_loss, tred_loss, twod_proj_loss, self_consistency_loss, get_weighted_loss

_ITER_TAG = 'cam2cam'


def _center_to_pelvis(keypoints_2d, pelvis_i=6):
    """ pelvis -> (0, 0) """

    n_joints = keypoints_2d.shape[2]
    pelvis_point = keypoints_2d[:, :, pelvis_i, :]

    keypoints_2d = keypoints_2d - pelvis_point.unsqueeze(2).repeat(1, 1, n_joints, 1)  # in each view: joint coords - pelvis coords

    return keypoints_2d


def _normalize_per_view(keypoints_2d):
    """ pelvis -> (0, 0), corners -> (1, 1) """

    keypoints_2d = _center_to_pelvis(keypoints_2d)
    frobenius_norm = torch.norm(keypoints_2d, p='fro', dim=(2, 3))

    batch_size, n_views = keypoints_2d.shape[0], keypoints_2d.shape[1]

    # "divided by its Frobenius norm in the preprocessing"
    keypoints_2d = torch.cat([
        torch.cat([
            (keypoints_2d[batch_i, view_i] / frobenius_norm[batch_i, view_i]).unsqueeze(0)
            for view_i in range(n_views)
        ]).unsqueeze(0)
        for batch_i in range(batch_size)
    ])

    return keypoints_2d


def _get_cam2cam_gt(cameras):
    n_views = len(cameras)
    batch_size = len(cameras[0])

    pairs = np.uint8([
        [
            (i, j)
            for j in range(n_views)
        ]
        for i in range(n_views)
    ]).reshape(-1, 2)

    cam2cam_gts = torch.zeros(batch_size, len(pairs), 4, 4)
    for batch_i in range(batch_size):
        # GT roto-translation: (3 x 4 + last row [0, 0, 0, 1]) = [ [ rot2rot | trans2trans ], [0, 0, 0, 1] ]
        cam2cam_gts[batch_i] = torch.cat([
            torch.matmul(
                torch.FloatTensor(cameras[j][batch_i].extrinsics_padded),
                torch.inverse(torch.FloatTensor(cameras[i][batch_i].extrinsics_padded))
            ).unsqueeze(0)  # 1 x 4 x 4
            for (i, j) in pairs
        ])  # ~ (len(pairs), 4, 4)

    return cam2cam_gts.cuda(), pairs  # ~ (batch_size=8, len(pairs), 3, 3)


def _forward_cam2cam(cam2cam_model, detections, pairs, scale_trans2trans=1e3, gts=None):
    batch_size = detections.shape[0]
    cam2cam_preds = torch.empty(batch_size, len(pairs), 4, 4)

    for batch_i in range(batch_size):
        rot2rot, trans2trans = cam2cam_model(
            detections[batch_i]  # ~ (len(pairs), 2, n_joints=17, 2D)
        )
        trans2trans = trans2trans * scale_trans2trans

        if not (gts is None):  # GTs have been provided => use them
            rot2rot = gts[batch_i, :, :3, :3].cuda().detach().clone()
            # no noise rot2rot = rot2rot + 0.1 * torch.rand_like(rot2rot)

            trans2trans = gts[batch_i, :, :3, 3].cuda().detach().clone()
            # no noise trans2trans = trans2trans + 1e2 * torch.rand_like(trans2trans)

        trans2trans = trans2trans.unsqueeze(0).view(len(pairs), 3, 1)  # .T

        for pair_i in range(len(pairs)):
            R = rot2rot[pair_i]
            t = trans2trans[pair_i]
            extrinsic = torch.cat([  # `torch.hstack`, for compatibility with cluster
                R, t
            ], dim=1)

            cam2cam_preds[batch_i, pair_i] = torch.cat([  # `torch.vstack`, for compatibility with cluster
                extrinsic,
                torch.cuda.FloatTensor([0, 0, 0, 1]).unsqueeze(0)
            ], dim=0)  # add [0, 0, 0, 1] at the bottom -> 4 x 4

    return cam2cam_preds.cuda()


def _do_dlt(cam2cam_preds, keypoints_2d_pred, confidences_pred, cameras, master_cam_i=0):
    batch_size = keypoints_2d_pred.shape[0]
    n_joints = keypoints_2d_pred.shape[2]

    keypoints_3d_pred = torch.zeros(
        batch_size,
        n_joints,
        3
    )

    for batch_i in range(batch_size):
        master_cam = cameras[master_cam_i][batch_i]
        target_cams_i = [1, 2, 3]  # todo use `master_cam`
        target_cams = [
            cameras[cam_i][batch_i]
            for cam_i in target_cams_i
        ]

        full_cam2cams = torch.cat([
            torch.cuda.FloatTensor(master_cam.intrinsics_padded).unsqueeze(0),
            torch.mm(
                torch.cuda.FloatTensor(target_cams[0].intrinsics_padded),
                cam2cam_preds[batch_i, master_cam_i, target_cams_i[0]]
            ).unsqueeze(0),
            torch.mm(
                torch.cuda.FloatTensor(target_cams[1].intrinsics_padded),
                cam2cam_preds[batch_i, master_cam_i, target_cams_i[1]]
            ).unsqueeze(0),
            torch.mm(
                torch.cuda.FloatTensor(target_cams[2].intrinsics_padded),
                cam2cam_preds[batch_i, master_cam_i, target_cams_i[2]]
            ).unsqueeze(0),
        ])  # ~ 4, 3, 4

        # ... perform DLT in master cam space, but since ...
        keypoints_3d_pred[batch_i] = triangulate_points_in_camspace(
            keypoints_2d_pred[batch_i],
            full_cam2cams,
            confidences_batch=confidences_pred[batch_i]
        )

    # ... they're in master cam space => cam2world
    return torch.cat([
        cameras[master_cam_i][batch_i].cam2world()(
            keypoints_3d_pred[batch_i]
        ).unsqueeze(0)
        for batch_i in range(batch_size)
    ])


def _compute_losses(cam2cam_preds, cam2cam_gts, keypoints_2d_pred, keypoints_3d_pred, keypoints_3d_gt, keypoints_3d_binary_validity_gt, cameras, config):
    _pairs = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]  # todo use `master_cam`

    total_loss = 0.0  # real loss, the one grad is applied to

    roto_loss = L2_R_loss(cam2cam_gts, cam2cam_preds, _pairs)
    if config.cam2cam.loss.roto_weight > 0:
        total_loss += config.cam2cam.loss.roto_weight * roto_loss

    geodesic_loss = geo_loss(cam2cam_gts, cam2cam_preds, _pairs)
    if config.cam2cam.loss.geo_weight > 0:
        total_loss += config.cam2cam.loss.geo_weight * geodesic_loss

    trans_loss = t_loss(
        cam2cam_gts, cam2cam_preds, _pairs, config.cam2cam.scale_trans2trans
    )
    if config.cam2cam.loss.trans_weight > 0:
        total_loss += config.cam2cam.loss.trans_weight * trans_loss

    pose_loss = twod_proj_loss(
        keypoints_3d_gt, keypoints_3d_pred, cameras
    )
    if config.cam2cam.loss.proj_weight > 0:
        total_loss += config.cam2cam.loss.proj_weight * pose_loss

    batch_size = keypoints_3d_pred.shape[0]
    master_cam_i = 0
    keypoints_cam_pred = torch.cat([
        torch.matmul(
            euclidean_to_homogeneous(
                keypoints_3d_pred[batch_i]  # [x y z] -> [x y z 1]
            ),
            torch.FloatTensor(cameras[master_cam_i][batch_i].extrinsics.T)
        ).unsqueeze(0)
        for batch_i in range(batch_size)
    ])
    loss_R, loss_t, loss_proj = self_consistency_loss(
        cameras, keypoints_cam_pred, cam2cam_preds, keypoints_2d_pred, config.cam2cam.scale_trans2trans / 1e1
    )

    # todo https://www.healthline.com/health/unexplained-weight-loss

    if config.cam2cam.loss.self_consistency.R > 0:
        total_loss += get_weighted_loss(
            loss_R, config.cam2cam.loss.self_consistency.R, 0.5, 2.0
        )
    if config.cam2cam.loss.self_consistency.t > 0:
        total_loss += get_weighted_loss(
            loss_t, config.cam2cam.loss.self_consistency.t, 0.5, 1.5
        )
    if config.cam2cam.loss.self_consistency.proj > 0:
        total_loss += get_weighted_loss(
            loss_proj, config.cam2cam.loss.self_consistency.proj, 1e1, 4e4
        )

    scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0
    loss_3d = tred_loss(
        keypoints_3d_gt,
        keypoints_3d_pred,
        keypoints_3d_binary_validity_gt,
        scale_keypoints_3d
    )
    if config.cam2cam.loss.tred_weight > 0:
        total_loss += get_weighted_loss(
            loss_3d, config.cam2cam.loss.tred_weight, 1e1, 7e2
        )

    return geodesic_loss, trans_loss, pose_loss, roto_loss, loss_3d, loss_R, loss_t, loss_proj, total_loss


def batch_iter(batch, iter_i, dataloader, model, cam2cam_model, _, opt, scheduler, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon):
    batch_size, n_views = images_batch.shape[0], images_batch.shape[1]
    n_joints = config.model.backbone.num_joints

    def _forward_kp():
        if config.cam2cam.using_gt:
            return get_kp_gt(keypoints_3d_gt, batch['cameras'])
        else:
            return model(
                images_batch, None, minimon
            )

    def _prepare_cam2cam_heatmaps_batch(heatmaps, pairs):
        heatmap_w, heatmap_h = heatmaps.shape[-2], heatmaps.shape[-1]
        detections = torch.empty(
            batch_size, len(pairs), 2, n_joints, heatmap_w, heatmap_h
        )

        for batch_i in range(batch_size):
            detections[batch_i] = torch.cat([
                torch.cat([
                    heatmaps[batch_i, i].unsqueeze(0),  # ~ (1, 17, 32, 32)
                    heatmaps[batch_i, j].unsqueeze(0)
                ]).unsqueeze(0)  # ~ (1, 2, 17, 32, 32)
                for (i, j) in pairs
            ])  # ~ (3, 2, 17, 32, 32)

        return detections.cuda()

    def _prepare_cam2cam_keypoints_batch(keypoints, pairs):
        detections = torch.empty(batch_size, len(pairs), 2, n_joints, 2)

        for batch_i in range(batch_size):
            detections[batch_i] = torch.cat([
                torch.cat([
                    keypoints[batch_i, i].unsqueeze(0),  # ~ (1, 17, 2)
                    keypoints[batch_i, j].unsqueeze(0)
                ]).unsqueeze(0)  # ~ (1, 2, 17, 2)
                for (i, j) in pairs
            ])  # ~ (3, 2, 17, 2)

        return detections.cuda()

    def _backprop():
        geodesic_loss, trans_loss, pose_loss, _, loss_3d, loss_R, loss_t, loss_proj, total_loss = _compute_losses(
            cam2cam_preds,
            cam2cam_gts,
            keypoints_2d_pred,
            keypoints_3d_pred,
            keypoints_3d_gt,
            keypoints_3d_binary_validity_gt,
            batch['cameras'],
            config
        )

        message = '{} batch iter {:d} losses: GEO ~ {:.3f}, TRANS ~ {:.3f}, POSE ~ {:.3f}, 3D ~ {:.3f}, SELF R ~ {:.3f}, SELF t ~ {:.3f}, SELF P ~ {:.3f}, TOTAL ~ {:.3f}'.format(
            'training' if is_train else 'validation',
            iter_i,
            geodesic_loss.item(),  # normalize per each sample
            trans_loss.item(),
            pose_loss.item(),
            loss_3d.item(),
            loss_R.item(),
            loss_t.item(),
            loss_proj.item(),
            total_loss.item(),
        )
        live_debug_log(_ITER_TAG, message)

        scalar_metric, _ = dataloader.dataset.evaluate(
            keypoints_3d_pred.detach().cpu().numpy(),
            batch['indexes'],
            split_by_subject=True
        )  # MPJPE
        message = '{} batch iter {:d} MPJPE: ~ {:.3f} mm'.format(
            'training' if is_train else 'validation',
            iter_i,
            scalar_metric
        )
        live_debug_log(_ITER_TAG, message)

        minimon.enter()

        current_lr = opt.param_groups[0]['lr']
        clip = config.cam2cam.opt.grad_clip / current_lr
        backprop(
            opt, total_loss, scheduler, scalar_metric, _ITER_TAG,
            get_grad_params(cam2cam_model), clip
        )
        minimon.leave('backward pass')

    minimon.enter()
    keypoints_2d_pred, heatmaps_pred, confidences_pred = _forward_kp()
    minimon.leave('BB forward')

    cam2cam_gts, pairs = _get_cam2cam_gt(batch['cameras'])
    if config.cam2cam.using_heatmaps:
        detections = _prepare_cam2cam_heatmaps_batch(heatmaps_pred, pairs)
    else:
        if config.cam2cam.normalize_kp_to_pelvis:
            kps = _normalize_per_view(keypoints_2d_pred)
            detections = _prepare_cam2cam_keypoints_batch(kps, pairs)
        else:
            detections = _prepare_cam2cam_keypoints_batch(keypoints_2d_pred, pairs)

    minimon.enter()
    cam2cam_preds = _forward_cam2cam(
        cam2cam_model,
        detections,
        pairs,
        config.cam2cam.scale_trans2trans,
        #cam2cam_gts
    )

    cam2cam_preds = cam2cam_preds.view(batch_size, n_views, n_views, 4, 4)
    cam2cam_gts = cam2cam_gts.view(batch_size, n_views, n_views, 4, 4)
    minimon.leave('cam2cam forward')

    minimon.enter()
    keypoints_3d_pred = _do_dlt(
        cam2cam_preds,
        keypoints_2d_pred,
        confidences_pred,
        batch['cameras']
    )
    minimon.leave('cam2cam DLT')

    if is_train:
        _backprop()

    return keypoints_3d_pred.detach()
