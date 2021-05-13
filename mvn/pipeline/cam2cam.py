import torch

from mvn.utils.misc import live_debug_log
from mvn.utils.multiview import triangulate_batch_of_points_in_cam_space
from mvn.datasets.utils import cam2cam_precomputed_batch
from mvn.models.loss import KeypointsMSESmoothLoss
from mvn.models.layers import R6DBlock


def batch_iter(batch, iter_i, model, cam2cam_model, criterion, opt, scheduler, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon):
    _iter_tag = 'cam2cam'

    scale_trans2trans = config.cam2cam.scale_trans2trans  # L2 loss on trans vectors is poor -> scale
    batch_size, n_views = images_batch.shape[0], images_batch.shape[1]
    n_joints = config.model.backbone.num_joints  # todo infer

    minimon.enter()

    if config.cam2cam.using_gt:
        # too verbose live_debug_log(_iter_tag, 'I\'m using GT 2D keypoints')

        keypoints_2d_pred = torch.cat([
            torch.cat([
                batch['cameras'][view_i][batch_i].world2proj()(
                    keypoints_3d_gt[batch_i].detach().cpu()  # ~ (17, 3)
                ).unsqueeze(0)
                for view_i in range(n_views)
            ]).unsqueeze(0)
            for batch_i in range(batch_size)
        ])  # ~ (batch_size, n_views, 17, 2)
        keypoints_2d_pred.requires_grad = False

        heatmaps_pred = torch.zeros(
            (batch_size, n_views, n_joints, 32, 32)
        )  # todo fake heatmaps_pred from GT KP: ~ N
        heatmaps_pred.requires_grad = False

        confidences_pred = torch.ones(
            (batch_size, n_views, n_joints)
        )  # 100% confident in each view
    else:
        keypoints_2d_pred, heatmaps_pred, confidences_pred = model(
            images_batch, None, minimon
        )

    minimon.leave('BB forward pass')

    # prepare GTs (cam2cam) and KP for forward
    pairs = [(0, 1), (0, 2), (0, 3)]  # 0 cam will be the "master"
    cam2cam_gts = torch.zeros(batch_size, len(pairs), 4, 4)

    if config.cam2cam.using_heatmaps:
        heatmap_w, heatmap_h = heatmaps_pred.shape[-2], heatmaps_pred.shape[-1]
        keypoints_forward = torch.empty(
            batch_size, len(pairs), 2, n_joints, heatmap_w, heatmap_h
        )
    else:
        keypoints_forward = torch.empty(batch_size, len(pairs), 2, n_joints, 2)

    for batch_i in range(batch_size):
        if config.cam2cam.using_heatmaps:
            keypoints_forward[batch_i] = torch.cat([
                torch.cat([
                    heatmaps_pred[batch_i, i].unsqueeze(0),  # ~ (1, 17, 32, 32)
                    heatmaps_pred[batch_i, j].unsqueeze(0)
                ]).unsqueeze(0)  # ~ (1, 2, 17, 32, 32)
                for (i, j) in pairs
            ])  # ~ (3, 2, 17, 32, 32)
        else:
            keypoints_forward[batch_i] = torch.cat([
                torch.cat([
                    keypoints_2d_pred[batch_i, i].unsqueeze(0),  # ~ (1, 17, 2)
                    keypoints_2d_pred[batch_i, j].unsqueeze(0)
                ]).unsqueeze(0)  # ~ (1, 2, 17, 2)
                for (i, j) in pairs
            ])  # ~ (3, 2, 17, 2)

        # GT roto-translation: (3 x 4 + last row [0, 0, 0, 1]) = [ [ rot2rot | trans2trans ], [0, 0, 0, 1] ]
        cam2cam_gts[batch_i] = torch.cat([
            torch.matmul(
                torch.FloatTensor(batch['cameras'][j][batch_i].extrinsics_padded),
                torch.inverse(torch.FloatTensor(batch['cameras'][i][batch_i].extrinsics_padded))
            ).unsqueeze(0)  # 1 x 4 x 4
            for (i, j) in pairs
        ])  # ~ (len(pairs), 4, 4)

    cam2cam_gts = cam2cam_gts.cuda()  # ~ (batch_size=8, len(pairs), 3, 3)
    keypoints_forward = keypoints_forward.cuda()

    minimon.enter()

    cam2cam_preds = torch.empty(batch_size, len(pairs), 4, 4)

    for batch_i in range(batch_size):
        rot2rot, trans2trans = cam2cam_model(
            keypoints_forward[batch_i]  # ~ (len(pairs), 2, n_joints=17, 2D)
        )
        trans2trans *= scale_trans2trans

        # todo GT
        # rot2rot = cam2cam_gts[batch_i, :, :3, :3].cuda().detach().clone()
        # trans2trans = cam2cam_gts[batch_i, :, :3, 3].cuda().detach().clone()

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

    minimon.leave('cam2cam forward')

    minimon.enter()

    # reconstruct full cam2cam and ...
    full_cam2cam = [
        [None] + list(cam2cam_preds[batch_i].cuda())
        for batch_i in range(batch_size)
    ]  # `None` for cam_0 -> cam_0
    
    # todo random master cam
    full_cam2cam = torch.cat([
        torch.cat([
            cam2cam_precomputed_batch(
                0, view_i, batch['cameras'], batch_i, full_cam2cam
            ).unsqueeze(0)
            for view_i in range(n_views)
        ]).unsqueeze(0)
        for batch_i in range(batch_size)
    ])

    # ... perform DLT in master cam space, but since ...
    keypoints_3d_pred = triangulate_batch_of_points_in_cam_space(
        full_cam2cam.cpu(),
        keypoints_2d_pred.cpu(),
        confidences_batch=confidences_pred.cpu()
    )

    # ... they're in master cam space => cam2world
    keypoints_3d_pred = torch.cat([
        batch['cameras'][0][batch_i].cam2world()(
            keypoints_3d_pred[batch_i]
        ).unsqueeze(0)
        for batch_i in range(batch_size)
    ])

    minimon.leave('cam2cam DLT')

    if is_train:
        # not exactly needed, just to debug
        roto_loss = 0.0
        geodesic_loss = 0.0
        trans_loss = 0.0
        pose_loss = 0.0
        loss_3d = 0.0
        total_loss = 0.0  # real loss, the one grad is applied to

        minimon.enter()

        for batch_i in range(batch_size):  # foreach sample in batch
            # L2 loss on rotation matrix
            loss = KeypointsMSESmoothLoss(threshold=0.5)(
                cam2cam_preds[batch_i, :, :3, :3].cuda(),
                cam2cam_gts[batch_i, :, :3, :3].cuda()
            )  # ~ (len(pairs), )
            roto_loss += loss
            if config.cam2cam.loss.roto_weight > 0:
                total_loss += config.cam2cam.loss.roto_weight * loss

            # geodesic loss on rotation matrix
            loss = R6DBlock.compute_geodesic_distance(
                cam2cam_preds[batch_i, :, :3, :3].cuda(),
                cam2cam_gts[batch_i, :, :3, :3].cuda()
            )  # ~ (len(pairs), )
            geodesic_loss += loss
            if config.cam2cam.loss.geo_weight > 0:
                total_loss += config.cam2cam.loss.geo_weight * loss

            # trans loss
            loss = KeypointsMSESmoothLoss(threshold=400)(
                cam2cam_preds[batch_i, :, :3, 3].cuda() / scale_trans2trans,
                cam2cam_gts[batch_i, :, :3, 3].cuda() / scale_trans2trans
            )
            trans_loss += loss
            if config.cam2cam.loss.trans_weight > 0:
                total_loss += config.cam2cam.loss.trans_weight * loss

            # 3D KP in world loss
            scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0

            loss = criterion(
                keypoints_3d_pred.cuda() * scale_keypoints_3d,  # ~ 8, 17, 3
                keypoints_3d_gt * scale_keypoints_3d,  # ~ 8, 17, 3
                keypoints_3d_binary_validity_gt.cuda()  # ~ 8, 17, 1
            )
            loss_3d += loss
            if config.cam2cam.loss.tred_weight > 0:
                total_loss += config.cam2cam.loss.tred_weight * loss

            # 2D KP projections loss, as in https://arxiv.org/abs/1905.10711
            gts = torch.cat([
                batch['cameras'][view_i][batch_i].world2proj()(
                    keypoints_3d_gt[batch_i]
                ).unsqueeze(0)
                for view_i in range(1, n_views)  # 0 is "master" cam
            ])  # ~ n_views - 1, 17, 2
            preds = torch.cat([
                batch['cameras'][view_i][batch_i].world2proj()(
                    keypoints_3d_pred[batch_i]
                ).unsqueeze(0)
                for view_i in range(1, n_views)  # 0 is "master" cam
            ])  # ~ n_views - 1, 17, 2
            loss = KeypointsMSESmoothLoss(threshold=400)(
                gts.cuda(),
                preds.cuda(),
            )
            pose_loss += loss
            if config.cam2cam.loss.proj_weight > 0:
                total_loss += config.cam2cam.loss.proj_weight * loss

        minimon.leave('calc loss')

        message = '{} batch iter {:d} avg per sample loss: GEO ~ {:.3f}, TRANS ~ {:.3f}, POSE ~ {:.3f}, ROTO ~ {:.3f}, 3D ~ {:.3f}, TOTAL ~ {:.3f}'.format(
            'training' if is_train else 'validation',
            iter_i,
            geodesic_loss.item() / batch_size,  # normalize per each sample
            trans_loss.item() / batch_size,
            pose_loss.item() / batch_size,
            roto_loss.item() / batch_size,
            loss_3d.item() / batch_size,
            total_loss.item() / batch_size,
        )  # just a little bit of live debug
        live_debug_log(_iter_tag, message)

        minimon.enter()

        opt.zero_grad()

        try:
            total_loss.backward()  # backward foreach batch
        except:
            live_debug_log(
                _iter_tag,
                'cannot backpropagate ... are you cheating?'
            )

        if hasattr(config.opt, "grad_clip"):  # clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.opt.grad_clip / config.opt.lr
            )

        opt.step()
        scheduler.step(total_loss.item() / batch_size)

        minimon.leave('backward pass')

    return keypoints_3d_pred.detach()
