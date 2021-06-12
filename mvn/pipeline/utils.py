import torch

from mvn.utils.misc import live_debug_log


# todo refactor
def get_kp_gt(keypoints_3d_gt, cameras, noisy=False):
    batch_size, n_joints, n_views = keypoints_3d_gt.shape[0], keypoints_3d_gt.shape[1], len(cameras)

    keypoints_2d_pred = torch.cat([
        torch.cat([
            cameras[view_i][batch_i].world2proj()(
                keypoints_3d_gt[batch_i].detach().cpu()  # ~ (17, 3)
            ).unsqueeze(0)
            for view_i in range(n_views)
        ]).unsqueeze(0)
        for batch_i in range(batch_size)
    ])  # ~ (batch_size, n_views, 17, 2)

    if noisy:  # todo batched
        for batch_i in range(batch_size):
            var = 0.2  # to be scaled with K
            for view_i in range(n_views):
                for joint_i in range(n_joints):
                    keypoints_2d_pred[batch_i, view_i, joint_i] +=\
                        torch.randn_like(
                            keypoints_2d_pred[batch_i, view_i, joint_i]
                        ) * var

    keypoints_2d_pred.requires_grad = False

    heatmaps_pred = torch.zeros(
        (batch_size, n_views, n_joints, 32, 32)
    )  # todo fake heatmaps_pred from GT KP: ~ N
    heatmaps_pred.requires_grad = False

    confidences_pred = torch.ones(
        (batch_size, n_views, n_joints), requires_grad=False
    )  # 100% confident in each view

    return keypoints_2d_pred, heatmaps_pred, confidences_pred


def backprop(opt, total_loss, scheduler, scheduler_metric, tag, params, clip):
    opt.zero_grad()

    try:
        total_loss.backward()  # backward foreach batch
    except:
        live_debug_log(
            tag,
            'cannot backpropagate ... are you cheating?'
        )

    if clip > 0.0:  # see #16578951: works well at the start, but then it stills
        torch.nn.utils.clip_grad_norm_(
            params,
            clip
        )

    opt.step()
    scheduler.step(scheduler_metric)
