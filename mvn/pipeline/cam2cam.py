import os

import torch

from mvn.models.utils import get_grad_params
from mvn.pipeline.utils import get_kp_gt, backprop
from mvn.utils.misc import live_debug_log, get_master_pairs
from mvn.utils.multiview import triangulate_batch_of_points_in_cam_space, euclidean_to_homogeneous, homogeneous_to_euclidean
from mvn.models.loss import GeodesicLoss, MSESmoothLoss, KeypointsMSESmoothLoss, ProjectionLoss, SeparationLoss, ScaleIndependentProjectionLoss, HuberLoss, WorldStructureLoss
from mvn.utils.tred import matrix_to_euler_angles, euler_angles_to_matrix

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


def _get_cams_gt(cameras, master_i=0):
    n_cameras = len(cameras)
    batch_size = len(cameras[0])
    cameras_i = get_master_pairs()

    cam_gts = torch.zeros(batch_size, n_cameras, 4, 4)
    for batch_i in range(batch_size):
        cam_gts[batch_i] = torch.cat([
            torch.DoubleTensor(cameras[camera_i][batch_i].extrinsics_padded).unsqueeze(0)
            for camera_i in cameras_i[master_i]
        ])  # ~ (| pairs |, 4, 4)

    return cam_gts.cuda()


def _forward_cams(cam2cam_model, detections, scale_t, gt=None, noisy=False):
    batch_size = detections.shape[0]
    n_views = detections.shape[1]

    preds = cam2cam_model(
        detections  # ~ (batch_size, | pairs |, 2, n_joints=17, 2D)
    )  # (batch_size, | pairs |, 3, 3)
    dev = preds.device

    for batch_i in range(batch_size):  # todo batched
        for view_i in range(n_views):
            # apply post processing ... todo as module

            # ... scale distance
            preds[batch_i, view_i, :3, 3] = preds[batch_i, view_i, :3, 3] * scale_t

            if not (gt is None):
                R = gt[batch_i, view_i, :3, :3].to(dev).detach().clone()
                t = gt[batch_i, view_i, :3, 3].to(dev).detach().clone()

                if noisy:  # noisy
                    R = R + 1e-1 * torch.rand_like(R)
                    t = t + 1e2 * torch.rand_like(t)

                preds[batch_i, view_i, :3, :3] = R
                preds[batch_i, view_i, :3, 3] = t

    return preds.to(dev)


def _prepare_cams_for_dlt(cams, keypoints_2d_pred, same_K_for_all):
    batch_size = keypoints_2d_pred.shape[0]
    full_cams = torch.empty((batch_size, 4, 3, 4))

    for batch_i in range(batch_size):
        from_master_cam = torch.inverse(cams[batch_i, 0])  # master is first
        full_cams[batch_i] = torch.cat([
            same_K_for_all.unsqueeze(0),  # doing DLT  in camspace
            torch.mm(
                same_K_for_all,
                torch.mm(
                    cams[batch_i, 1],
                    from_master_cam
                )
            ).unsqueeze(0),
            torch.mm(
                same_K_for_all,
                torch.mm(
                    cams[batch_i, 2],
                    from_master_cam
                )
            ).unsqueeze(0),
            torch.mm(
                same_K_for_all,
                torch.mm(
                    cams[batch_i, 3],
                    from_master_cam
                )
            ).unsqueeze(0),
        ])  # ~ 4, 3, 4

    return full_cams  # ~ batch_size, 4, 3, 4


def triangulate(cams, keypoints_2d_pred, confidences_pred, K, master_cam_i):
    full_cams = _prepare_cams_for_dlt(
        cams,
        keypoints_2d_pred,
        K
    )

    # ... perform DLT in master cam space, but since ...
    kps_cam_pred = triangulate_batch_of_points_in_cam_space(
        full_cams.cpu(),
        keypoints_2d_pred.cpu(),
        confidences_batch=confidences_pred.cpu()
    )

    # ... they're in master cam space => cam2world
    batch_size = cams.shape[0]
    return torch.cat([
        homogeneous_to_euclidean(
            euclidean_to_homogeneous(
                kps_cam_pred[batch_i]
            ).to(cams.device)
            @
            torch.inverse(
                cams[batch_i, master_cam_i].T
            )
        ).unsqueeze(0)
        for batch_i in range(batch_size)
    ])


def _compute_losses(cam_preds, cam_gts, keypoints_2d_pred, kps_world_pred, kps_world_gt, keypoints_3d_binary_validity_gt, cameras, config):
    dev = cam_preds.device
    total_loss = torch.tensor(0.0).to(dev)  # real loss, the one grad is applied to
    batch_size = cam_preds.shape[0]
    n_cameras = cam_preds.shape[1]

    loss_R = GeodesicLoss()(
        cam_gts.view(batch_size * n_cameras, 4, 4)[:, :3, :3],  # just R
        cam_preds.view(batch_size * n_cameras, 4, 4)[:, :3, :3]
    )
    if config.cam2cam.loss.R > 0:
        total_loss += config.cam2cam.loss.R * loss_R

    trans_loss = MSESmoothLoss(threshold=4e2)(
        cam_gts.view(batch_size * n_cameras, 4, 4)[:, :3, 3] / config.cam2cam.postprocess.scale_t,  # just t
        cam_preds.view(batch_size * n_cameras, 4, 4)[:, :3, 3] / config.cam2cam.postprocess.scale_t,
    )
    if config.cam2cam.loss.t > 0:
        total_loss += config.cam2cam.loss.t * trans_loss

    K = torch.DoubleTensor(cameras[0][0].intrinsics_padded)  # same for all
    loss_proj = ProjectionLoss()(
        kps_world_gt,
        kps_world_pred,
        cameras,
        K,
        cam_preds
    )
    if config.cam2cam.loss.proj > 0:
        total_loss += config.cam2cam.loss.proj * loss_proj

    loss_self_proj = ScaleIndependentProjectionLoss(HuberLoss(threshold=1e-1))(
        K,
        cam_preds,
        kps_world_pred,
        keypoints_2d_pred
    ) * 1e3  # final scaling
    if config.cam2cam.loss.self_consistency.proj > 0:
        total_loss += loss_self_proj * config.cam2cam.loss.self_consistency.proj

    loss_self_separation = SeparationLoss(3e1)(kps_world_pred)
    if config.cam2cam.loss.self_consistency.separation > 0:
        total_loss += loss_self_separation * config.cam2cam.loss.self_consistency.separation

    loss_world_structure = WorldStructureLoss(1e2)(cam_preds)
    if config.cam2cam.loss.world_structure.camera_above_surface > 0:
        total_loss += loss_world_structure * config.cam2cam.loss.world_structure.camera_above_surface

    # todo ? loss_body_structure = BodyStructureLoss(250)(kps_world_pred)

    __batch_i = 0  # todo debug only

    print('pred exts {:.0f}'.format(__batch_i))
    print(cam_preds[__batch_i])
    print('gt exts {:.0f}'.format(__batch_i))
    print(cam_gts[__batch_i])

    print('pred batch {:.0f}'.format(__batch_i))
    print(kps_world_pred[__batch_i])
    print('gt batch {:.0f}'.format(__batch_i))
    print(kps_world_gt[__batch_i])

    loss_world = KeypointsMSESmoothLoss(threshold=20*20)(
        kps_world_pred * config.opt.scale_keypoints_3d,
        kps_world_gt.to(dev) * config.opt.scale_keypoints_3d,
        keypoints_3d_binary_validity_gt,
    )
    if config.cam2cam.loss.world > 0:
        total_loss += loss_world * config.cam2cam.loss.world

    return loss_R, trans_loss,\
        loss_proj, loss_world,\
        loss_self_proj, loss_self_separation, \
        loss_world_structure, \
        total_loss


def batch_iter(epoch_i, batch, iter_i, model, cam2cam_model, opt, scheduler, images_batch, kps_world_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon, experiment_dir):
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
            keypoints_2d_pred, heatmaps_pred, confidences_pred = get_kp_gt(kps_world_gt, batch['cameras'], config.cam2cam.data.using_noise)
        else:
            keypoints_2d_pred, heatmaps_pred, confidences_pred = model(
                images_batch, None, minimon
            )

        return keypoints_2d_pred, heatmaps_pred, confidences_pred

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
        loss_R, trans_loss, loss_2d, loss_3d, loss_self_proj, loss_self_separation, loss_world_structure, total_loss = _compute_losses(
            cam_preds,
            cam_gts,
            keypoints_2d_pred,
            kps_world_pred,
            kps_world_gt,
            keypoints_3d_binary_validity_gt,
            batch['cameras'],
            config
        )

        # assuming to have access to 1 GT world point
        # joint_i = 9  # head
        # loss_pose_ref = KeypointsMSESmoothLoss(threshold=20*20)(
        #     kps_world_pred[:, joint_i] * config.opt.scale_keypoints_3d,
        #     kps_world_gt[:, joint_i].to(kps_world_pred.device) * config.opt.scale_keypoints_3d,
        #     keypoints_3d_binary_validity_gt[:, joint_i],  # HEAD only
        # )
        # total_loss += 5.0 * loss_pose_ref

        message = '{} batch iter {:d} losses: R ~ {:.1f}, t ~ {:.1f}, 2D ~ {:.0f}, 3D ~ {:.0f}, SELF 2D ~ {:.0f}, SELF SEP ~ {:.0f}, WORLD STRUCT ~ {:.0f}, TOTAL ~ {:.0f}'.format(
            'training' if is_train else 'validation',
            iter_i,
            loss_R.item(),
            trans_loss.item(),
            loss_2d.item(),
            loss_3d.item(),
            loss_self_proj.item(),
            loss_self_separation.item(),
            loss_world_structure.item(),
            total_loss.item(),
        )
        live_debug_log(_ITER_TAG, message)

        minimon.enter()

        current_lr = opt.param_groups[0]['lr']
        clip = config.cam2cam.opt.grad_clip / current_lr

        backprop(
            opt, total_loss, scheduler,
            loss_self_proj * 1.1,  # give some slack
            _ITER_TAG, get_grad_params(cam2cam_model), clip
        )
        minimon.leave('backward pass')

    minimon.enter()
    keypoints_2d_pred, _, confidences_pred = _forward_kp()
    if config.debug.dump_tensors:
        _save_stuff(keypoints_2d_pred, 'keypoints_2d_pred')
    minimon.leave('BB forward')
    print(keypoints_2d_pred[0])

    cam_gts = _get_cams_gt(batch['cameras'])
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

    master_i = 0  # views are randomly sorted => no need for a random master within batch
    cam_preds = _forward_cams(
        cam2cam_model,
        detections[master_i],
        config.cam2cam.postprocess.scale_t,
        cam_gts if config.debug.gt_cams else None,
        noisy=config.debug.noisy
    )
    if config.debug.dump_tensors:
        _save_stuff(cam_preds, 'cam_preds')

    minimon.leave('cam2cam forward')

    minimon.enter()

    ordered = get_master_pairs()[master_i]
    kps_world_pred = triangulate(
        cam_preds,
        keypoints_2d_pred[:, ordered],
        confidences_pred,
        torch.cuda.DoubleTensor(batch['cameras'][0][0].intrinsics_padded),
        master_i
    )
    if config.debug.dump_tensors:
        _save_stuff(kps_world_pred, 'kps_world_pred')
        _save_stuff(batch['indexes'], 'batch_indexes')
        _save_stuff(kps_world_gt, 'kps_world_gt')

    minimon.leave('cam2cam DLT')

    if is_train:
        _backprop()

    return kps_world_pred.detach().cpu()  # no need for grad no more
