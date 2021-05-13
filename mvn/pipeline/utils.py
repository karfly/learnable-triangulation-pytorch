import torch

from mvn.utils.misc import live_debug_log


def get_kp_gt(keypoints_3d_gt, cameras):
    batch_size, n_joints, n_views = keypoints_3d_gt.shape[0], keypoints_3d_gt.shape[1], 4  # todo infer

    keypoints_2d_pred = torch.cat([
        torch.cat([
            cameras[view_i][batch_i].world2proj()(
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

    return keypoints_2d_pred, heatmaps_pred, confidences_pred


def backprop(opt, scheduler, total_loss, tag):
    opt.zero_grad()

    try:
        total_loss.backward()  # backward foreach batch
    except:
        live_debug_log(
            tag,
            'cannot backpropagate ... are you cheating?'
        )

    opt.step()
    scheduler.step(total_loss.item())
