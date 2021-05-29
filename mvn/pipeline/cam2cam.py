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
                    dist2pelvis(kps[batch_i, view_i])
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
    cameras_i = get_master_pairs()

    cam2cam_gts = torch.zeros(n_cameras, batch_size, n_cameras, 4, 4)
    for master_i in range(n_cameras):
        for batch_i in range(batch_size):
            cam2cam_gts[master_i, batch_i] = torch.cat([
                torch.DoubleTensor(cameras[camera_i][batch_i].extrinsics_padded).unsqueeze(0)
                for camera_i in cameras_i[master_i]
            ])  # ~ (| pairs |, 4, 4)

    return cam2cam_gts.cuda()


def _forward_cam2cam(cam2cam_model, detections, master_i, scale_t, gt=None, noisy=False):
    batch_size = detections[master_i].shape[0]
    n_views = detections[master_i].shape[1]

    preds = cam2cam_model(
        detections[master_i]  # ~ (batch_size, | pairs |, 2, n_joints=17, 2D)
    )  # (batch_size, | pairs |, 3, 3)

    for batch_i in range(batch_size):  # todo batched
        for view_i in range(n_views):
            # apply post processing
            preds[batch_i, view_i, :3, 3] = preds[batch_i, view_i, :3, 3] * scale_t

            if not (gt is None):
                R = gt[master_i, batch_i, view_i, :3, :3].cuda().detach().clone()
                t = gt[master_i, batch_i, view_i, :3, 3].cuda().detach().clone()

                if noisy:  # noisy
                    R = R + 1e-1 * torch.rand_like(R)
                    t = t + 1e2 * torch.rand_like(t)

                preds[batch_i, view_i, :3, :3] = R
                preds[batch_i, view_i, :3, 3] = t

    return preds.cuda()


def _prepare_cams_for_dlt(cams, keypoints_2d_pred, cameras, master_cam_i):
    batch_size = keypoints_2d_pred.shape[0]
    full_cams = torch.empty((batch_size, 4, 3, 4))

    for batch_i in range(batch_size):
        master_cam = cameras[master_cam_i][batch_i]

        # full_cams[batch_i] = torch.cat([
        #     torch.cuda.DoubleTensor(master_cam.intrinsics_padded).unsqueeze(0),
        #     torch.mm(
        #         torch.cuda.DoubleTensor(target_cams[0].intrinsics_padded),
        #         torch.mm(
        #             cams[batch_i, 1],
        #             torch.inverse(cams[batch_i, master_cam_i])
        #         )
        #     ).unsqueeze(0),
        #     torch.mm(
        #         torch.cuda.DoubleTensor(target_cams[1].intrinsics_padded),
        #         torch.mm(
        #             cams[batch_i, 2],
        #             torch.inverse(cams[batch_i, master_cam_i])
        #         )
        #     ).unsqueeze(0),
        #     torch.mm(
        #         torch.cuda.DoubleTensor(target_cams[2].intrinsics_padded),
        #         torch.mm(
        #             cams[batch_i, 3],
        #             torch.inverse(cams[batch_i, master_cam_i])
        #         )
        #     ).unsqueeze(0),
        # ])  # ~ 4, 3, 4

        same_K_for_all = torch.cuda.DoubleTensor(master_cam.intrinsics_padded)
        full_cams[batch_i] = torch.cat([
            torch.mm(
                same_K_for_all,
                cams[batch_i, 0]  # the master is the first
            ).unsqueeze(0),
            torch.mm(
                same_K_for_all,
                cams[batch_i, 1]
            ).unsqueeze(0),
            torch.mm(
                same_K_for_all,
                cams[batch_i, 2],
            ).unsqueeze(0),
            torch.mm(
                same_K_for_all,
                cams[batch_i, 3],
            ).unsqueeze(0),
        ])  # ~ 4, 3, 4

    return full_cams  # ~ batch_size, 4, 3, 4


def _do_dlt(cam2cams, keypoints_2d_pred, confidences_pred, cameras, master_cam_i):
    full_cam2cams = _prepare_cams_for_dlt(
        cam2cams, keypoints_2d_pred, cameras, master_cam_i
    )

    # ... perform DLT in master cam space, but since ...
    kps_cam_pred = triangulate_batch_of_points_in_cam_space(
        full_cam2cams.cpu(),
        keypoints_2d_pred.cpu(),
        confidences_batch=confidences_pred.cpu()
    )
    return kps_cam_pred

    # ... they're in master cam space => cam2world
    # return torch.cat([
    #     homogeneous_to_euclidean(
    #         euclidean_to_homogeneous(
    #             kps_cam_pred[batch_i]
    #         ).to(cam2cams.device)
    #         @
    #         torch.inverse(
    #             cam2cams[batch_i, master_cam_i].T
    #         )
    #     ).unsqueeze(0)
    #     for batch_i in range(batch_size)
    # ])


def _compute_losses(master2other_preds, cam2cam_gts, keypoints_2d_pred, kps_world_pred, kps_world_gt, keypoints_3d_binary_validity_gt, cameras, config):
    total_loss = 0.0  # real loss, the one grad is applied to

    master_cam_i = 0
    geodesic_loss = geo_loss(
        cam2cam_gts[master_cam_i],
        master2other_preds[master_cam_i]
    )
    if config.cam2cam.loss.R > 0:
        total_loss += config.cam2cam.loss.R * geodesic_loss

    trans_loss = t_loss(
        cam2cam_gts[master_cam_i],
        master2other_preds[master_cam_i],
        config.cam2cam.postprocess.scale_t
    )
    if config.cam2cam.loss.t > 0:
        total_loss += config.cam2cam.loss.t * trans_loss

    loss_proj = twod_proj_loss(
        kps_world_gt,
        kps_world_pred,
        cameras,
        master2other_preds[master_cam_i]
    )
    if config.cam2cam.loss.proj > 0:
        total_loss += config.cam2cam.loss.proj * loss_proj

    loss_self_cam, loss_self_proj, loss_self_world = self_consistency_loss(
        cameras,
        master2other_preds,
        kps_world_pred,
        keypoints_2d_pred,
        master_cam_i,
        config.cam2cam.postprocess.scale_t,
        config.opt.scale_keypoints_3d
    )
    if config.cam2cam.loss.self_consistency.cam > 0:
        total_loss += get_weighted_loss(
            loss_self_cam, config.cam2cam.loss.self_consistency.cam, 1e1, 4e4
        )
    if config.cam2cam.loss.self_consistency.proj > 0:
        total_loss += get_weighted_loss(
            loss_self_proj, config.cam2cam.loss.self_consistency.proj, 1e1, 4e4
        )
    if config.cam2cam.loss.self_consistency.world > 0:
        total_loss += get_weighted_loss(
            loss_self_world, config.cam2cam.loss.self_consistency.world, 1e1, 4e4
        )

    loss_world = tred_loss(
        kps_world_pred,  # all masters are equivalent => mean
        kps_world_gt,
        keypoints_3d_binary_validity_gt,
        config.opt.scale_keypoints_3d
    )
    if config.cam2cam.loss.world > 0:
        total_loss += get_weighted_loss(
            loss_world, config.cam2cam.loss.world, 1e1, 7e2
        )

    return geodesic_loss, trans_loss,\
        loss_proj, loss_world,\
        loss_self_cam, loss_self_proj, loss_self_world,\
        total_loss


def batch_iter(epoch_i, batch, iter_i, dataloader, model, cam2cam_model, _, opt, scheduler, images_batch, kps_world_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon, experiment_dir):
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
            return get_kp_gt(kps_world_gt, batch['cameras'])
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
        geodesic_loss, trans_loss, loss_2d, loss_3d, loss_self_cam, loss_self_2d, loss_self_world, total_loss = _compute_losses(
            cam_preds,
            cam2cam_gts,
            keypoints_2d_pred,
            kps_world_pred,
            kps_world_gt,
            keypoints_3d_binary_validity_gt,
            batch['cameras'],
            config
        )

        message = '{} batch iter {:d} losses: R ~ {:.1f}, t ~ {:.1f}, 2D ~ {:.0f}, 3D ~ {:.0f}, SELF CAM ~ {:.0f}, SELF 2D ~ {:.0f}, SELF 3D ~ {:.0f}, TOTAL ~ {:.0f}'.format(
            'training' if is_train else 'validation',
            iter_i,
            geodesic_loss.item(),
            trans_loss.item(),
            loss_2d.item(),
            loss_3d.item(),
            loss_self_cam.item(),
            loss_self_2d.item(),
            loss_self_world.item(),
            total_loss.item(),
        )
        live_debug_log(_ITER_TAG, message)

        per_pose_error_relative, _, _ = dataloader.dataset.evaluate(
            kps_world_pred.detach().cpu().numpy(),
            batch['indexes'],
            split_by_subject=True
        )  # MPJPE
        message = '{} batch iter {:d} MPJPE: ~ {:.1f} mm'.format(
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
    keypoints_2d_pred, _, confidences_pred = _forward_kp()
    if config.debug.dump_tensors:
        _save_stuff(keypoints_2d_pred, 'keypoints_2d_pred')
    minimon.leave('BB forward')

    cam2cam_gts = _get_cam2cam_gt(batch['cameras'])
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
    master_i = np.random.randint(0, n_cameras)
    cam_preds = torch.cat([
        _forward_cam2cam(
            cam2cam_model,
            detections,
            master_i,
            config.cam2cam.postprocess.scale_t,
            cam2cam_gts if config.debug.gt_cams else None,
            noisy=config.debug.noisy
        ).unsqueeze(0)
        for master_i in range(n_cameras)  # forward all cam2cam for self.losses
    ])
    if config.debug.dump_tensors:
        _save_stuff(cam_preds, 'cam_preds')

    minimon.leave('cam2cam forward')

    minimon.enter()
    # kps_world_pred = torch.cat([
    ordered = get_master_pairs()[master_i]
    kps_world_pred = _do_dlt(
        cam_preds[master_i],
        keypoints_2d_pred[:, ordered],
        confidences_pred,
        batch['cameras'],
        master_i
    )
    #     for master_i, ordered in enumerate(get_master_pairs())
    # ])  # ~ n_master_cams, batch_size, (17 x 3D points)

    if config.debug.dump_tensors:
        _save_stuff(kps_world_pred, 'kps_world_pred')
        _save_stuff(batch['indexes'], 'batch_indexes')
        _save_stuff(kps_world_gt, 'kps_world_gt')

    minimon.leave('cam2cam DLT')

    if is_train:
        _backprop()

    return kps_world_pred.detach().cpu()
