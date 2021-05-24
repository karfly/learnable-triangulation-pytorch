from mvn.models.utils import get_grad_params
import torch

import numpy as np

from mvn.pipeline.utils import get_kp_gt, backprop
from mvn.utils.misc import live_debug_log, get_pairs, get_master_pairs
from mvn.utils.multiview import triangulate_batch_of_points_in_cam_space, euclidean_to_homogeneous
from mvn.models.loss import geo_loss, t_loss, tred_loss, twod_proj_loss, self_consistency_loss, get_weighted_loss

_ITER_TAG = 'cam2cam'


def _center_to_pelvis(keypoints_2d, pelvis_i=6):
    """ pelvis -> (0, 0) """

    n_joints = keypoints_2d.shape[2]
    pelvis_point = keypoints_2d[:, :, pelvis_i, :]

    return keypoints_2d - pelvis_point.unsqueeze(2).repeat(1, 1, n_joints, 1)  # in each view: joint coords - pelvis coords


def _normalize_fro_kps(keypoints_2d):
    """ "divided by its Frobenius norm in the preprocessing" """

    batch_size, n_views = keypoints_2d.shape[0], keypoints_2d.shape[1]
    kps = _center_to_pelvis(keypoints_2d)

    return torch.cat([
        torch.cat([
            (
                kps[batch_i, view_i] / 
                torch.norm(kps[batch_i, view_i], p='fro')
            ).unsqueeze(0)
            for view_i in range(n_views)
        ]).unsqueeze(0)
        for batch_i in range(batch_size)
    ])


def _normalize_kps(keypoints_2d):
    batch_size, n_views = keypoints_2d.shape[0], keypoints_2d.shape[1]

    return torch.cat([
        torch.cat([
            (
                (
                    keypoints_2d[batch_i, view_i] -
                    keypoints_2d[batch_i, view_i].mean(axis=0)
                ) / 1.0  # todo std?
            ).unsqueeze(0)
            for view_i in range(n_views)
        ]).unsqueeze(0)
        for batch_i in range(batch_size)
    ])


def _get_cam2cam_gt(cameras):
    batch_size = len(cameras[0])
    pairs = get_pairs(len(cameras))

    cam2cam_gts = torch.zeros(batch_size, len(pairs), 4, 4)
    for batch_i in range(batch_size):
        # GT roto-translation: (3 x 4 + last row [0, 0, 0, 1]) = [ [ rot2rot | trans2trans ], [0, 0, 0, 1] ]
        cam2cam_gts[batch_i] = torch.cat([
            torch.matmul(
                torch.DoubleTensor(cameras[j][batch_i].extrinsics_padded),
                torch.DoubleTensor(
                    np.linalg.inv(cameras[i][batch_i].extrinsics_padded)
                )
            ).unsqueeze(0)  # 1 x 4 x 4
            for (i, j) in pairs
        ])  # ~ (_normalize_kps, 4, 4)

    return cam2cam_gts.cuda(), pairs  # ~ (batch_size=8, | pairs |, 4, 4)


def _forward_cam2cam(cam2cam_model, detections, scale_trans2trans=1e3):
    batch_size = detections.shape[0]
    n_views = 4  # todo infer
    preds = torch.empty(batch_size, n_views - 1, 4, 4)

    rot2rot, trans2trans = cam2cam_model(
        detections  # ~ (*, | pairs |, 2, n_joints=17, 2D)
    )  # (batch_size, | pairs |, 3, 3), (batch_size, | pairs |, 3)

    for batch_i in range(batch_size):  # todo batched
        for pair_i in range(n_views - 1):
            R = rot2rot[batch_i, pair_i]
            t = trans2trans[batch_i, pair_i].unsqueeze(0).T * scale_trans2trans

            # todo use gt
            # rot2rot = gts[:, :, :3, :3].cuda().detach().clone()
            # trans2trans = gts[:, :, :3, 3].cuda().detach().clone() / scale_trans2trans

            # if False:  # noisy
            #     rot2rot = rot2rot + 0.1 * torch.rand_like(rot2rot)
            #     trans2trans = trans2trans + 1e2 * torch.rand_like(trans2trans)


            extrinsic = torch.cat([  # `torch.hstack`, for compatibility with cluster
                R, t.view(3, 1)
            ], dim=1)

            preds[batch_i, pair_i] = torch.cat([  # `torch.vstack`, for compatibility with cluster
                extrinsic,
                torch.cuda.DoubleTensor([0, 0, 0, 1]).unsqueeze(0)
            ], dim=0)  # add [0, 0, 0, 1] at the bottom -> 4 x 4

    return preds.cuda()


def _prepare_cam2cams_for_dlt(master2other_preds, keypoints_2d_pred, cameras, master_cam_i=0):
    batch_size = keypoints_2d_pred.shape[0]
    full_cam2cams = torch.empty((batch_size, 4, 3, 4))

    for batch_i in range(batch_size):
        master_cam = cameras[master_cam_i][batch_i]
        target_cams_i = [1, 2, 3]  # todo use `master_cam`
        target_cams = [
            cameras[cam_i][batch_i]
            for cam_i in target_cams_i
        ]

        full_cam2cams[batch_i] = torch.cat([
            torch.cuda.DoubleTensor(master_cam.intrinsics_padded).unsqueeze(0),
            torch.mm(
                torch.cuda.DoubleTensor(target_cams[0].intrinsics_padded),
                master2other_preds[batch_i, target_cams_i[0] - 1]
            ).unsqueeze(0),
            torch.mm(
                torch.cuda.DoubleTensor(target_cams[1].intrinsics_padded),
                master2other_preds[batch_i, target_cams_i[1] - 1]
            ).unsqueeze(0),
            torch.mm(
                torch.cuda.DoubleTensor(target_cams[2].intrinsics_padded),
                master2other_preds[batch_i, target_cams_i[2] - 1]
            ).unsqueeze(0),
        ])  # ~ 4, 3, 4

    print(full_cam2cams.shape)

    return full_cam2cams


def _do_dlt(master2other_preds, keypoints_2d_pred, confidences_pred, cameras, master_cam_i=0):
    batch_size = keypoints_2d_pred.shape[0]

    full_cam2cams = _prepare_cam2cams_for_dlt(master2other_preds, keypoints_2d_pred, cameras)

    #print(keypoints_2d_pred)

    # ... perform DLT in master cam space, but since ...
    keypoints_3d_pred = triangulate_batch_of_points_in_cam_space(
        full_cam2cams.cpu(),
        keypoints_2d_pred,
        confidences_batch=confidences_pred
    )

    # ... they're in master cam space => cam2world
    return torch.cat([
        cameras[master_cam_i][batch_i].cam2world()(
            keypoints_3d_pred[batch_i]
        ).unsqueeze(0)
        for batch_i in range(batch_size)
    ])


def _compute_losses(master2other_preds, cam2cam_gts, keypoints_2d_pred, keypoints_3d_pred, keypoints_3d_gt, keypoints_3d_binary_validity_gt, cameras, config):
    _pairs = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]  # todo use `master_cam`

    total_loss = 0.0  # real loss, the one grad is applied to

    geodesic_loss = geo_loss(cam2cam_gts, master2other_preds, _pairs)
    if config.cam2cam.loss.geo_weight > 0:
        total_loss += config.cam2cam.loss.geo_weight * geodesic_loss

    trans_loss = t_loss(
        cam2cam_gts, master2other_preds, _pairs, config.cam2cam.scale_trans2trans
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
            torch.DoubleTensor(cameras[master_cam_i][batch_i].extrinsics.T)
        ).unsqueeze(0)
        for batch_i in range(batch_size)
    ])
    loss_R, loss_t, loss_proj = self_consistency_loss(
        cameras, keypoints_cam_pred, master2other_preds, keypoints_2d_pred, config.cam2cam.scale_trans2trans / 1e1  # original scale is too much ..
    )

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

    return geodesic_loss, trans_loss, pose_loss, loss_3d, loss_R, loss_t, loss_proj, total_loss


def batch_iter(epoch_i, batch, iter_i, dataloader, model, cam2cam_model, _, opt, scheduler, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon):
    batch_size = images_batch.shape[0]
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

    def _prepare_cam2cam_keypoints_batch(keypoints):
        """ master-cam KPs will be first, then the others """

        n_views = keypoints.shape[1]
        out = torch.zeros(batch_size, n_views, n_views, n_joints, 2)
        pairs = get_master_pairs()

        for batch_i in range(batch_size):  # todo batched
            for master_cam in range(n_views):
                out[batch_i, master_cam] = torch.cat([
                    keypoints[batch_i, view_i].unsqueeze(0)
                    for view_i in pairs[master_cam]
                ])

        return out.cuda()

    def _backprop():
        geodesic_loss, trans_loss, pose_loss, loss_3d, loss_R, loss_t, loss_proj, total_loss = _compute_losses(
            master2other_preds,
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
        n_batch_per_epoch = 2  # todo find with config
        n_iters = epoch_i * n_batch_per_epoch
        if n_iters < 4e2:
            clip = config.cam2cam.opt.grad_clip / current_lr
        else:  # gradient boost
            clip = config.cam2cam.opt.grad_clip * 2.0 / current_lr

        backprop(
            opt, total_loss, scheduler,
            scalar_metric + 10,  # give some slack (mm)
            _ITER_TAG, get_grad_params(cam2cam_model), clip
        )
        minimon.leave('backward pass')

    minimon.enter()
    keypoints_2d_pred, heatmaps_pred, confidences_pred = _forward_kp()
    minimon.leave('BB forward')

    cam2cam_gts, pairs = _get_cam2cam_gt(batch['cameras'])
    if config.cam2cam.using_heatmaps:
        detections = _prepare_cam2cam_heatmaps_batch(heatmaps_pred, pairs)
    else:
        if hasattr(config.cam2cam, 'normalize_kps'):
            if config.cam2cam.normalize_kps == 'fro':
                kps = _normalize_fro_kps(keypoints_2d_pred)
            elif config.cam2cam.normalize_kps == 'mean':
                kps = _normalize_kps(keypoints_2d_pred)

            detections = _prepare_cam2cam_keypoints_batch(kps)
        else:
            detections = _prepare_cam2cam_keypoints_batch(
                keypoints_2d_pred, pairs
            )

    minimon.enter()

    # forward all cam2cam for self.losses
    n_cameras = 4  # todo infer
    cam2cam_preds = torch.cat([
        _forward_cam2cam(
            cam2cam_model,
            detections[:, master_i, ...],
            config.cam2cam.scale_trans2trans
        ).unsqueeze(0)
        for master_i in range(n_cameras)
    ])

    master2other_preds = cam2cam_preds[0, ...].view(batch_size, 3, 4, 4)  # 0 is 'master'
    cam2cam_gts = cam2cam_gts.view(batch_size, 3, 4, 4)
    minimon.leave('cam2cam forward')

    minimon.enter()
    keypoints_3d_pred = _do_dlt(
        master2other_preds,
        keypoints_2d_pred,
        confidences_pred,
        batch['cameras'],
    )
    minimon.leave('cam2cam DLT')

    if is_train:
        _backprop()

    return keypoints_3d_pred.detach()
