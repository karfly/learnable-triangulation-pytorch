import os
from mvn.models.utils import get_grad_params
import torch

import numpy as np

from mvn.pipeline.utils import get_kp_gt, backprop
from mvn.utils.misc import live_debug_log, get_others, get_pairs, get_master_pairs
from mvn.utils.multiview import triangulate_batch_of_points_in_cam_space, euclidean_to_homogeneous, homogeneous_to_euclidean
from mvn.models.loss import geo_loss, t_loss, tred_loss, twod_proj_loss, self_consistency_loss, get_weighted_loss

_ITER_TAG = 'cam2cam'


def _center_to_pelvis(keypoints_2d, pelvis_i=6):
    """ pelvis -> (0, 0) """

    n_joints = keypoints_2d.shape[2]
    pelvis_point = keypoints_2d[:, :, pelvis_i, :]

    return keypoints_2d - pelvis_point.unsqueeze(2).repeat(1, 1, n_joints, 1)  # in each view: joint coords - pelvis coords


def dist2pelvis(keypoints_2d_in_view, pelvis_i=6):
    return torch.mean(torch.cat([
        torch.norm(
            keypoints_2d_in_view[i] -\
            keypoints_2d_in_view[pelvis_i]
        ).unsqueeze(0)
        for i in range(keypoints_2d_in_view.shape[0])
        if i != pelvis_i
    ])).unsqueeze(0)


def _normalize_keypoints(keypoints_2d, pelvis_center_kps, normalization):
    """ "divided by its Frobenius norm in the preprocessing" """

    batch_size, n_views = keypoints_2d.shape[0], keypoints_2d.shape[1]

    if pelvis_center_kps:
        kps = _center_to_pelvis(keypoints_2d)
    else:
        kps = keypoints_2d

    if normalization == 'd2pelvis':
        scaling = torch.cat([
            torch.max(
                torch.cat([
                    dist2pelvis(kps[batch_i, view_i]) * 1e1
                    for view_i in range(n_views)
                ]).unsqueeze(0)
            ).unsqueeze(0).repeat(1, n_views)  # same for each view
            for batch_i in range(batch_size)
        ])
    elif normalization == 'fro':
        scaling = torch.cat([
            torch.cat([
                torch.norm(kps[batch_i, view_i], p='fro').unsqueeze(0)
                for view_i in range(n_views)
            ]).unsqueeze(0)
            for batch_i in range(batch_size)
        ])
    elif normalization == 'maxfro':
        scaling = torch.cat([
            torch.max(
                torch.cat([
                    torch.norm(kps[batch_i, view_i], p='fro').unsqueeze(0)
                    for view_i in range(n_views)
                ]).unsqueeze(0)
            ).unsqueeze(0).repeat(1, n_views)  # same for each view
            for batch_i in range(batch_size)
        ])

    return torch.cat([
        torch.cat([
            (
                kps[batch_i, view_i] / scaling[batch_i, view_i]
            ).unsqueeze(0)
            for view_i in range(n_views)
        ]).unsqueeze(0)
        for batch_i in range(batch_size)
    ])


def _get_cam2cam_gt(cameras):
    n_cameras = len(cameras)
    batch_size = len(cameras[0])
    pairs_per_master = get_pairs()
    
    # cam2cam_gts = torch.zeros(, batch_size, n_cameras - 1, 4, 4)
    # for master_i in range(n_cameras):
    #     pairs = pairs_per_master[master_i]
    #     for batch_i in range(batch_size):
    #         cam2cam_gts[master_i, batch_i] = torch.cat([
    #             torch.mm(
    #                 torch.DoubleTensor(cameras[j][batch_i].extrinsics_padded),
    #                 torch.DoubleTensor(
    #                     np.linalg.inv(cameras[i][batch_i].extrinsics_padded)
    #                 )
    #             ).unsqueeze(0)  # 1 x 4 x 4
    #             for (i, j) in pairs
    #         ])  # ~ (| pairs |, 4, 4)

    cam2cam_gts = torch.zeros(batch_size, n_cameras, 4, 4)
    for batch_i in range(batch_size):
        cam2cam_gts[batch_i] = torch.cat([
            torch.DoubleTensor(
                cameras[i][batch_i].extrinsics_padded
            ).unsqueeze(0)  # 1 x 4 x 4
            for i in range(n_cameras)
        ])

    return cam2cam_gts.cuda(), pairs_per_master


def _forward_cam2cam(cam2cam_model, detections, scale_trans2trans, gt=None, noisy=False):
    batch_size = detections.shape[0]
    n_views = detections.shape[1]

    preds = cam2cam_model(
        detections  # ~ (batch_size, | pairs |, 2, n_joints=17, 2D)
    )  # (batch_size, | pairs |, 3, 3)

    for batch_i in range(batch_size):  # todo batched
        for view_i in range(n_views):
        # for pair_i in range(n_views - 1):
            preds[batch_i, view_i, :3, 3] = preds[batch_i, view_i, :3, 3] * scale_trans2trans
            
            if not (gt is None):
                R = gt[batch_i, view_i, :3, :3].cuda().detach().clone()
                t = gt[batch_i, view_i, :3, 3].cuda().detach().clone()

                if noisy:  # noisy
                    R = R + 1e-2 * torch.rand_like(R)
                    t = t + 1e2 * torch.rand_like(t)

                preds[batch_i, view_i, :3, :3] = R
                preds[batch_i, view_i, :3, 3] = t

    return preds.cuda()


def _prepare_cam2cams_for_dlt(cam2cams, keypoints_2d_pred, cameras, master_cam_i):
    batch_size = keypoints_2d_pred.shape[0]
    full_cam2cams = torch.empty((batch_size, 4, 3, 4))
    target_cams_i = get_others()[master_cam_i]

    for batch_i in range(batch_size):
        master_cam = cameras[master_cam_i][batch_i]
        target_cams = [
            cameras[cam_i][batch_i]
            for cam_i in target_cams_i
        ]

        full_cam2cams[batch_i] = torch.cat([
            torch.cuda.DoubleTensor(master_cam.intrinsics_padded).unsqueeze(0),
            torch.mm(
                torch.cuda.DoubleTensor(target_cams[0].intrinsics_padded),
                torch.mm(
                    cam2cams[batch_i, 1],
                    torch.inverse(cam2cams[batch_i, master_cam_i])
                )
                # cam2cams[batch_i, 0]
            ).unsqueeze(0),
            torch.mm(
                torch.cuda.DoubleTensor(target_cams[1].intrinsics_padded),
                torch.mm(
                    cam2cams[batch_i, 2],
                    torch.inverse(cam2cams[batch_i, master_cam_i])
                )
                # cam2cams[batch_i, 1]
            ).unsqueeze(0),
            torch.mm(
                torch.cuda.DoubleTensor(target_cams[2].intrinsics_padded),
                torch.mm(
                    cam2cams[batch_i, 3],
                    torch.inverse(cam2cams[batch_i, master_cam_i])
                )
                # cam2cams[batch_i, 2]
            ).unsqueeze(0),
        ])  # ~ 4, 3, 4

    return full_cam2cams  # ~ batch_size, 4, 3, 4


def _do_dlt(cam2cams, keypoints_2d_pred, confidences_pred, cameras, master_cam_i):
    batch_size = keypoints_2d_pred.shape[0]

    full_cam2cams = _prepare_cam2cams_for_dlt(
        cam2cams, keypoints_2d_pred, cameras, master_cam_i
    )

    # ... perform DLT in master cam space, but since ...
    keypoints_3d_pred = triangulate_batch_of_points_in_cam_space(
        full_cam2cams.cpu(),
        keypoints_2d_pred,
        confidences_batch=confidences_pred
    )

    # ... they're in master cam space => cam2world
    masters2world = torch.cat([
        torch.inverse(
            torch.cat([
                cam2cams[batch_i, master_cam_i, :, :3],  # predicted R
                torch.DoubleTensor(
                    cameras[master_cam_i][batch_i].extrinsics_padded[:, 3]
                ).unsqueeze(-1).to(cam2cams.device)  # GT t
            ], dim=-1).T
        ).unsqueeze(0)
        for batch_i in range(batch_size)
    ])

    # return torch.cat([
    #     cameras[master_cam_i][batch_i].cam2world()(
    #         keypoints_3d_pred[batch_i]
    #     ).unsqueeze(0)
    #     for batch_i in range(batch_size)
    # ])

    return torch.cat([
        homogeneous_to_euclidean(
            euclidean_to_homogeneous(
                keypoints_3d_pred[batch_i]
            ).to(cam2cams.device)
            @
            # masters2world[batch_i]
            torch.inverse(
                torch.DoubleTensor(
                    cameras[master_cam_i][batch_i].extrinsics_padded.T
                )  # using GT
            ).to(cam2cams.device)
        ).unsqueeze(0)
        for batch_i in range(batch_size)
    ])


def _compute_losses(master2other_preds, cam2cam_gts, keypoints_2d_pred, keypoints_3d_pred, keypoints_3d_gt, keypoints_3d_binary_validity_gt, cameras, config):
    total_loss = 0.0  # real loss, the one grad is applied to

    geodesic_loss = geo_loss(cam2cam_gts, master2other_preds)
    if config.cam2cam.loss.geo_weight > 0:
        total_loss += config.cam2cam.loss.geo_weight * geodesic_loss

    trans_loss = t_loss(
        cam2cam_gts, master2other_preds, config.cam2cam.postprocess.scale_trans2trans
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
    # keypoints_cam_pred = torch.cat([
    #     torch.mm(
    #         euclidean_to_homogeneous(
    #             keypoints_3d_pred[batch_i]  # [x y z] -> [x y z 1]
    #         ),
    #         torch.DoubleTensor(cameras[master_cam_i][batch_i].extrinsics.T)
    #     ).unsqueeze(0)
    #     for batch_i in range(batch_size)
    # ])
    keypoints_cam_pred = None

    loss_ext, loss_proj = self_consistency_loss(
        cameras, master2other_preds, keypoints_cam_pred, keypoints_2d_pred, master_cam_i
    )
    if config.cam2cam.loss.self_consistency.ext > 0:
        total_loss += get_weighted_loss(
            loss_ext, config.cam2cam.loss.self_consistency.ext, 1.0, 5.0
        )
    if config.cam2cam.loss.self_consistency.proj > 0:
        total_loss += get_weighted_loss(
            loss_proj, config.cam2cam.loss.self_consistency.proj, 1e1, 4e4
        )

    loss_3d = tred_loss(
        keypoints_3d_gt,
        keypoints_3d_pred,
        keypoints_3d_binary_validity_gt,
        config.opt.scale_keypoints_3d
    )
    if config.cam2cam.loss.tred_weight > 0:
        total_loss += get_weighted_loss(
            loss_3d, config.cam2cam.loss.tred_weight, 1e1, 7e2
        )

    return geodesic_loss, trans_loss,\
        pose_loss, loss_3d,\
        loss_ext, loss_proj,\
        total_loss


def batch_iter(epoch_i, batch, iter_i, dataloader, model, cam2cam_model, _, opt, scheduler, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon, experiment_dir):
    batch_size = images_batch.shape[0]
    n_joints = config.model.backbone.num_joints
    iter_folder = 'epoch-{:.0f}-iter-{:.0f}'.format(epoch_i, iter_i)
    iter_dir = os.path.join(experiment_dir, iter_folder) if experiment_dir else None

    def _save_stuff(stuff, var_name):
        if iter_dir:
            os.makedirs(iter_dir, exist_ok=True)
            f_out = os.path.join(iter_dir, var_name + '.trc')
            torch.save(torch.tensor(stuff), f_out)

    def _forward_kp():
        if config.cam2cam.data.using_gt:
            return get_kp_gt(keypoints_3d_gt, batch['cameras'])
        else:
            return model(
                images_batch, None, minimon
            )

    def _prepare_cam2cam_keypoints_batch(keypoints):
        """ master-cam KPs will be first, then the others """

        n_views = keypoints.shape[1]
        out = torch.zeros(n_views, batch_size, n_views, n_joints, 2)
        pairs = get_master_pairs()

        for batch_i in range(batch_size):  # todo batched
            for master_cam in range(n_views):
                out[master_cam, batch_i] = torch.cat([
                    keypoints[batch_i, view_i].unsqueeze(0)
                    for view_i in pairs[master_cam]
                ])

        return out.cuda()

    def _backprop():
        geodesic_loss, trans_loss, pose_loss, loss_3d, loss_ext, loss_proj, total_loss = _compute_losses(
            cam2cam_preds,
            cam2cam_gts,
            keypoints_2d_pred,
            keypoints_3d_pred,
            keypoints_3d_gt,
            keypoints_3d_binary_validity_gt,
            batch['cameras'],
            config
        )

        message = '{} batch iter {:d} losses: GEO ~ {:.3f}, TRANS ~ {:.3f}, POSE ~ {:.3f}, 3D ~ {:.3f}, SELF ext ~ {:.3f}, SELF P ~ {:.3f}, TOTAL ~ {:.3f}'.format(
            'training' if is_train else 'validation',
            iter_i,
            geodesic_loss.item(),  # normalize per each sample
            trans_loss.item(),
            pose_loss.item(),
            loss_3d.item(),
            loss_ext.item(),
            loss_proj.item(),
            total_loss.item(),
        )
        live_debug_log(_ITER_TAG, message)

        per_pose_error_relative, per_pose_error_absolute, _ = dataloader.dataset.evaluate(
            keypoints_3d_pred.detach().cpu().numpy(),
            batch['indexes'],
            split_by_subject=True
        )  # MPJPE
        message = '{} batch iter {:d} MPJPE: ~ {:.3f} mm'.format(
            'training' if is_train else 'validation',
            iter_i,
            per_pose_error_relative
        )
        live_debug_log(_ITER_TAG, message)

        minimon.enter()

        current_lr = opt.param_groups[0]['lr']
        clip = config.cam2cam.opt.grad_clip / current_lr

        backprop(
            opt, total_loss, scheduler,
            per_pose_error_relative + 15,  # give some slack (mm)
            _ITER_TAG, get_grad_params(cam2cam_model), clip
        )
        minimon.leave('backward pass')

    minimon.enter()
    keypoints_2d_pred, heatmaps_pred, confidences_pred = _forward_kp()
    if config.debug.dump_tensors:
        _save_stuff(keypoints_2d_pred, 'keypoints_2d_pred')
    minimon.leave('BB forward')

    cam2cam_gts, _ = _get_cam2cam_gt(batch['cameras'])
    if config.cam2cam.data.using_heatmaps:
        pass  # todo detections = _prepare_cam2cam_heatmaps_batch(heatmaps_pred, pairs)
    else:
        if hasattr(config.cam2cam.preprocess, 'normalize_kps'):
            kps = _normalize_keypoints(
                keypoints_2d_pred,
                config.cam2cam.preprocess.pelvis_center_kps,
                config.cam2cam.preprocess.normalize_kps
            )

            detections = _prepare_cam2cam_keypoints_batch(kps)
            if config.debug.dump_tensors:
                _save_stuff(detections, 'detections')
        else:
            pass  # todo detections = _prepare_cam2cam_keypoints_batch

    minimon.enter()

    n_cameras = len(batch['cameras'])
    master_i = 0  # todo rand
    # cam2cam_preds = torch.cat([
    #     _forward_cam2cam(
    #         cam2cam_model,
    #         detections[master_i],
    #         config.cam2cam.postprocess.scale_trans2trans,
    #         # cam2cam_gts[master_i],
    #         noisy=config.debug.noisy
    #     ).unsqueeze(0)
    #     for master_i in range(n_cameras)  # forward all cam2cam for self.losses
    # ])
    cam2cam_preds = _forward_cam2cam(
        cam2cam_model,
        detections[master_i],
        config.cam2cam.postprocess.scale_trans2trans,
        cam2cam_gts if config.debug.gt_cams else None,
        noisy=config.debug.noisy
    )
    if config.debug.dump_tensors:
        _save_stuff(cam2cam_preds, 'cam2cam_preds')

    minimon.leave('cam2cam forward')

    minimon.enter()
    keypoints_3d_pred = _do_dlt(
        cam2cam_preds,
        keypoints_2d_pred,
        confidences_pred,
        batch['cameras'],
        master_i
    )

    if config.debug.dump_tensors:
        _save_stuff(keypoints_3d_pred, 'keypoints_3d_pred')
        _save_stuff(batch['indexes'], 'batch_indexes')
        _save_stuff(keypoints_3d_gt, 'keypoints_3d_gt')

    minimon.leave('cam2cam DLT')

    if is_train:
        _backprop()

    return keypoints_3d_pred.detach().cpu()
