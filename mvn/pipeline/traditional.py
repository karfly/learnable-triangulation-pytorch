from torch import nn

from mvn.models.loss import VolumetricCELoss


def batch_iter(batch, iter_i, model, model_type, criterion, opt, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, proj_matricies_batch, is_train, config, minimon):
    minimon.enter()

    if model_type == "alg" or model_type == "ransac":
        keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(
            images_batch,
            proj_matricies_batch,  # ~ (batch_size=8, n_views=4, 3, 4)
            minimon
        )  # keypoints_3d_pred, keypoints_2d_pred ~ (8, 17, 3), (~ 8, 4, 17, 2)
    elif model_type == "vol":
        keypoints_3d_pred, heatmaps_pred, volumes_pred, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred = model(
            images_batch,
            proj_matricies_batch,
            batch,
            minimon
        )

    minimon.leave('forward pass')

    if is_train:
        use_volumetric_ce_loss = config.opt.use_volumetric_ce_loss if hasattr(config.opt, "use_volumetric_ce_loss") else False

        minimon.enter()

        if config.opt.loss_2d:  # ~ 0 seconds
            batch_size, n_views = images_batch.shape[0], images_batch.shape[1]
            total_loss = 0.0

            for batch_i in range(batch_size):
                for view_i in range(n_views):  # todo faster (batched) loop
                    cam = batch['cameras'][view_i][batch_i]

                    gt = cam.world2proj()(keypoints_3d_gt[batch_i])  # ~ 17, 2
                    pred = cam.world2proj()(
                        keypoints_3d_pred[batch_i]
                    )  # ~ 17, 2

                    total_loss += criterion(
                        pred.unsqueeze(0).cuda(),  # ~ 1, 17, 2
                        gt.unsqueeze(0).cuda(),  # ~ 1, 17, 2
                        keypoints_3d_binary_validity_gt[batch_i].unsqueeze(0).cuda()  # ~ 1, 17, 1
                    )
        elif config.opt.loss_3d:  # ~ 0 seconds
            scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0

            total_loss = criterion(
                keypoints_3d_pred.cuda() * scale_keypoints_3d,  # ~ 8, 17, 3
                keypoints_3d_gt.cuda() * scale_keypoints_3d,  # ~ 8, 17, 3
                keypoints_3d_binary_validity_gt.cuda()  # ~ 8, 17, 1
            )
        elif use_volumetric_ce_loss:
            volumetric_ce_criterion = VolumetricCELoss()

            loss = volumetric_ce_criterion(
                coord_volumes_pred, volumes_pred, keypoints_3d_gt, keypoints_3d_binary_validity_gt
            )

            weight = config.opt.volumetric_ce_loss_weight if hasattr(config.opt, "volumetric_ce_loss_weight") else 1.0

            total_loss = weight * loss

        print('  {} batch iter {:d} loss ~ {:.3f}'.format(
            'training' if is_train else 'validation',
            iter_i,
            total_loss.item()
        ))  # just a little bit of live debug

        minimon.leave('calc loss')

        minimon.enter()

        opt.zero_grad()
        total_loss.backward()  # backward foreach batch

        if hasattr(config.opt, "grad_clip"):
            nn.utils.clip_grad_norm_(
                model.parameters(),
                config.opt.grad_clip / config.opt.lr
            )

        opt.step()

        minimon.leave('backward pass')

    return keypoints_3d_pred.detach().cpu().numpy()
