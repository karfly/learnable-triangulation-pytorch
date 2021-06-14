import torch
import numpy as np
from itertools import permutations

from mvn.models.rototrans import RotoTransCombiner
from mvn.utils.tred import euler_angles_to_matrix
from mvn.utils.multiview import _my_proj
from mvn.utils.misc import live_debug_log


# todo refactor
def get_kp_gt(keypoints_3d_gt, cameras, use_extra_cams=False, noisy=False):
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

    if use_extra_cams:
        eulers_per_axis = np.linspace(-np.pi * 0.9, np.pi * 0.9, 5) 
        eulers = np.float64(list(permutations(  # todo uniform
            eulers_per_axis, r=3
        )))  # todo random
        Rs = euler_angles_to_matrix(
            torch.tensor(eulers), 'XYZ'  # or any other
        )
        distances = np.linspace(4e3, 6e3, len(eulers))  # todo random
        Rts = RotoTransCombiner()(
            Rs.unsqueeze(0),  # batched ...
            torch.tensor(distances).view(1, len(eulers), 1)
        )[0]
        K = torch.tensor(cameras[0][0].intrinsics_padded)  # same for all
        
        fakes = torch.cat([
            torch.cat([
                _my_proj(Rts[fake_i], K)(
                    keypoints_3d_gt[batch_i].detach().cpu()  # ~ (17, 3)
                ).unsqueeze(0)
                for fake_i in range(len(Rts))
            ]).unsqueeze(0)
            for batch_i in range(batch_size)
        ])
        keypoints_2d_pred = torch.cat([
            keypoints_2d_pred,
            fakes,
        ], dim=1)  # ~ (batch_size, n_views + |eulers|, 17, 2)

    if noisy:  # todo batched
        var = 0.2  # to be scaled with K ...
        keypoints_2d_pred += torch.randn_like(keypoints_2d_pred) * var

    keypoints_2d_pred.requires_grad = False

    heatmaps_pred = torch.zeros(
        (batch_size, n_views, n_joints, 32, 32)
    )  # todo fake heatmaps_pred from GT KP: ~ N
    heatmaps_pred.requires_grad = False

    confidences_pred = torch.ones(
        (batch_size, keypoints_2d_pred.shape[1], n_joints), requires_grad=False
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
