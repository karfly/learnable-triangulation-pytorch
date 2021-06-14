import torch
import numpy as np

from mvn.datasets.utils import cam2cam_batch
from mvn.utils.multiview import euclidean_to_homogeneous
from mvn.utils.misc import live_debug_log


def batch_iter(batch, iter_i, model, model_type, criterion, opt, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon):
    _iter_tag = 'DLT in cam'

    batch_size, n_views = images_batch.shape[0], images_batch.shape[1]
    master_cams = np.random.randint(0, n_views, size=batch_size)  # choose random "master" cam foreach frame in batch
    proj_matricies_batch = torch.tensor([
        [
            cam2cam_batch(
                master_cams[batch_i], view_i, batch['cameras'], batch_i
            )
            for view_i in range(n_views)
        ]
        for batch_i in range(batch_size)
    ])

    minimon.enter()

    keypoints_3d_pred, _, _, _ = model(
        images_batch,
        proj_matricies_batch,  # ~ (batch_size=8, n_views=4, 3, 4)
        minimon
    )

    minimon.leave('BB forward pass')

    if is_train:
        minimon.enter()

        if config.opt.loss_3d:  # variant I: 3D loss on cam KP
            live_debug_log(_iter_tag, 'using variant I (3D loss)')

            scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0

            gt_in_cam = [
                batch['cameras'][master_cams[batch_i]][batch_i].world2cam()(
                    keypoints_3d_gt[batch_i].cpu()
                ).numpy()
                for batch_i in range(batch_size)
            ]

            total_loss = criterion(
                keypoints_3d_pred.cuda() * scale_keypoints_3d,  # ~ 8, 17, 3
                torch.tensor(gt_in_cam).cuda() * scale_keypoints_3d,  # ~ 8, 17, 3
                keypoints_3d_binary_validity_gt.cuda()  # ~ 8, 17, 1
            )  # "the loss is 3D pose difference between the obtained 3D pose from DLT and the 3D pose in the first camera space"
        else:  # variant II (2D loss on each view)
            live_debug_log(_iter_tag, 'using variant II (2D loss on each view)')
            total_loss = 0.0

            for batch_i in range(batch_size):
                master_cam = batch['cameras'][master_cams[batch_i]][batch_i]

                for view_i in range(n_views):  # todo faster loop
                    cam = batch['cameras'][view_i][batch_i]

                    gt = cam.world2proj()(keypoints_3d_gt[batch_i].detach().cpu())  # ~ 17, 2
                    pred = master_cam.cam2other(cam)(
                        euclidean_to_homogeneous(
                            keypoints_3d_pred[batch_i]
                        )
                    )  # ~ 17, 2

                    total_loss += criterion(
                        torch.tensor(pred).unsqueeze(0).cuda(),  # ~ 1, 17, 2
                        torch.tensor(gt).unsqueeze(0).cuda(),
                        keypoints_3d_binary_validity_gt[batch_i].unsqueeze(0).cuda()
                    )  # "The loss is then 2D pose difference between the 2D pose you obtain this way and the GT 2D pose in each view."

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
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.opt.grad_clip / config.opt.lr
            )

        opt.step()

        minimon.leave('backward pass')

    # they're in cam space => cam2world for metric evaluation
    keypoints_3d_pred = keypoints_3d_pred.detach()
    return torch.cat([
        batch['cameras'][master_cams[batch_i]][batch_i].cam2world()(
            keypoints_3d_pred[batch_i]
        ).unsqueeze(0)
        for batch_i in range(batch_size)
    ])
