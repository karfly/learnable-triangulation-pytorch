import numpy as np

import torch
from torch import nn

from mvn.utils.multiview import _2camspace, _2proj, _my_proj
from mvn.utils.misc import get_pairs, get_inverse_i_from_pair, get_master_pairs


class KeypointsMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsMSESmoothLoss(nn.Module):
    def __init__(self, threshold=20*20, alpha=0.1, beta=0.9):
        super().__init__()

        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity=None):
        dimension = keypoints_pred.shape[-1]
        diff = (keypoints_gt - keypoints_pred) ** 2

        if not (keypoints_binary_validity is None):
            diff *= keypoints_binary_validity

        diff[diff > self.threshold] = torch.pow(
            diff[diff > self.threshold], self.alpha
        ) * (self.threshold ** self.beta)  # soft version
        loss = torch.sum(diff) / dimension
        
        if not (keypoints_binary_validity is None):
            loss /= max(1, torch.sum(keypoints_binary_validity).item())

        return loss


class MSESmoothLoss(nn.Module):
    def __init__(self, threshold, alpha=0.1, beta=0.9):
        super().__init__()

        self.threshold = threshold

    def forward(self, pred, gt):
        diff = (gt - pred) ** 2

        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)  # soft version
        return torch.mean(diff)


class KeypointsMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum(torch.abs(keypoints_gt - keypoints_pred) * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        loss = torch.sum(torch.sqrt(torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity, dim=2)))
        loss = loss / max(1, torch.sum(keypoints_binary_validity).item())
        return loss


class VolumetricCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coord_volumes_batch, volumes_batch_pred, keypoints_gt, keypoints_binary_validity):
        loss = 0.0
        n_losses = 0

        batch_size = volumes_batch_pred.shape[0]
        for batch_i in range(batch_size):
            coord_volume = coord_volumes_batch[batch_i]
            keypoints_gt_i = keypoints_gt[batch_i]

            coord_volume_unsq = coord_volume.unsqueeze(0)
            keypoints_gt_i_unsq = keypoints_gt_i.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            dists = torch.sqrt(((coord_volume_unsq - keypoints_gt_i_unsq) ** 2).sum(-1))
            dists = dists.view(dists.shape[0], -1)

            min_indexes = torch.argmin(dists, dim=-1).detach().cpu().numpy()
            min_indexes = np.stack(np.unravel_index(min_indexes, volumes_batch_pred.shape[-3:]), axis=1)

            for joint_i, index in enumerate(min_indexes):
                validity = keypoints_binary_validity[batch_i, joint_i]
                loss += validity[0] * (-torch.log(volumes_batch_pred[batch_i, joint_i, index[0], index[1], index[2]] + 1e-6))
                n_losses += 1


        return loss / n_losses


def geodesic_distance(m1, m2):
    batch_size = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # ~ (batch_size, 3, 3)

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

    # bound [-1, 1]
    cos = torch.min(
        cos,
        torch.ones(batch_size).cuda()
    )
    cos = torch.max(
        cos,
        torch.ones(batch_size).cuda() * -1
    )

    theta = torch.acos(cos)

    return theta.mean()  # ~ (batch_size,)


def geo_loss(gts, preds, criterion=geodesic_distance):
    n_cameras = gts.shape[1]
    batch_size = gts.shape[0]

    return criterion(
        preds.view(batch_size * n_cameras, 4, 4)[:, :3, :3].cuda(),  # just R
        gts.view(batch_size * n_cameras, 4, 4)[:, :3, :3].cuda()
    )


def t_loss(gts, preds, scale_trans2trans, criterion=MSESmoothLoss(threshold=4e2)):
    dev = preds.device
    n_cameras = gts.shape[1]
    batch_size = gts.shape[0]

    return criterion(
        preds.view(batch_size * n_cameras, 4, 4)[:, :3, 3].to(dev) / scale_trans2trans,  # just t
        gts.view(batch_size * n_cameras, 4, 4)[:, :3, 3].to(dev) / scale_trans2trans
    )


def tred_loss(preds, gts, keypoints_3d_binary_validity_gt, scale_keypoints_3d, criterion=KeypointsMSESmoothLoss(threshold=20*20)):
    dev = preds.device
    return criterion(
        preds.to(dev) * scale_keypoints_3d,
        gts.to(dev) * scale_keypoints_3d,
        keypoints_3d_binary_validity_gt.to(dev)
    )


def twod_proj_loss(keypoints_3d_gt, keypoints_3d_pred, cameras, cam_preds, criterion=KeypointsMSESmoothLoss(threshold=20*20)):
    """ project GT VS pred points to all views """

    n_views = len(cameras)
    batch_size = keypoints_3d_gt.shape[0]
    dev = cam_preds.device
    return torch.mean(torch.cat([
        criterion(
            torch.cat([
                _my_proj(
                    cam_preds[batch_i, view_i],
                    torch.DoubleTensor(cameras[view_i][batch_i].intrinsics_padded).to(dev)
                )(keypoints_3d_pred[batch_i]).unsqueeze(0)
                for view_i in range(1, n_views)  # not considering master (0)
            ]).to(dev),  # pred
            torch.cat([
                cameras[view_i][batch_i].world2proj()(
                    keypoints_3d_gt[batch_i]
                ).unsqueeze(0)
                for view_i in range(1, n_views)  # 0 is "master" cam
            ]).to(dev),  # gt
        ).unsqueeze(0)
        for batch_i in range(batch_size)
    ]))


def _project_in_other_views(cameras, keypoints_mastercam_pred, cams_pred, master_cam_i):
    batch_size = len(cameras[0])
    pairs = get_pairs()[master_cam_i]
    pairs = [(0, 0)] + pairs  # project also to master
    dev = keypoints_mastercam_pred.device
    return torch.cat([
        torch.cat([
            _2proj(
                cams_pred[master_cam_i, batch_i, master_cam_i],
                cams_pred[master_cam_i, batch_i, target],
                torch.DoubleTensor(cameras[i][0].intrinsics_padded).to(dev),
            )(keypoints_mastercam_pred[batch_i]).unsqueeze(0)
            for i, (_, target) in enumerate(pairs)
        ]).unsqueeze(0)  # ~ n_views 1, 3, 17, 2
        for batch_i in range(batch_size)
    ])


def _self_consistency_cam2cam(cams_preds):
    ordered_views = get_master_pairs()
    n_cams = cams_preds.shape[0]
    batch_size = cams_preds.shape[1]
    dev = cams_preds.device

    loss_R = torch.tensor(0.0).to(dev)
    loss_t = torch.tensor(0.0).to(dev)

    for cam_i in range(n_cams):  # todo tensored
        index_cam = [
            master_views.index(cam_i)
            for master_views in ordered_views
        ]
        for batch_i in range(batch_size):  # todo tensored
            cams = torch.cat([
                cams_preds[row_i, batch_i, i].unsqueeze(0)
                for row_i, i in enumerate(index_cam)
            ])  # same extrinsics cam but predicted from a different master ...

            # ... should be the same
            Rs = cams[:, :3, :3]
            mean = torch.mean(Rs, axis=0).unsqueeze(0)
            loss_R += MSESmoothLoss(threshold=1e2)(Rs, mean)  # todo use geodesic

            ts = cams[:, 2, 3]
            mean = torch.mean(ts).unsqueeze(0)
            loss_t += MSESmoothLoss(threshold=1e2)(ts, mean)

    normalization = n_cams * batch_size

    loss_R = loss_R * 1e3 / normalization  # to compare against t
    loss_t = loss_t / normalization  # to compare against t

    return loss_R + loss_t


def _masochism(cameras, cams_preds, keypoints_cam_pred, initial_keypoints, master_cam_i, criterion=KeypointsMSESmoothLoss(threshold=20*20)):
    """ penalize trivial solutions """

    projections = _project_in_other_views(
        cameras, keypoints_cam_pred, cams_preds, master_cam_i
    )  # ~ 8, 3, 17, 2

    # todo



def _self_consistency_P(cameras, cams_preds, keypoints_cam_pred, initial_keypoints, master_cam_i, criterion=KeypointsMSESmoothLoss(threshold=20*20)):
    projections = _project_in_other_views(
        cameras, keypoints_cam_pred, cams_preds, master_cam_i
    )  # ~ 8, 3, 17, 2

    batch_size = len(cameras[0])
    pairs = get_pairs()[master_cam_i]
    pairs = [(0, 0)] + pairs  # project also to master

    print(projections[0])
    print(initial_keypoints[0])
    1/0

    return torch.mean(
        torch.cat([
            criterion(
                torch.cat([
                    initial_keypoints[batch_i, i].unsqueeze(0)
                    for _, i in pairs
                ]).to(keypoints_cam_pred.device),
                projections[batch_i].to(keypoints_cam_pred.device),
            ).unsqueeze(0)
            for batch_i in range(batch_size)
        ])
    )


def self_consistency_loss(cameras, cams_preds, keypoints_cam_pred, initial_keypoints, master_cam_i):
    loss_cam2cam = _self_consistency_cam2cam(cams_preds)
    loss_proj = _self_consistency_P(
        cameras, cams_preds, keypoints_cam_pred, initial_keypoints, master_cam_i
    )
    return loss_cam2cam, loss_proj  # todo and others


def get_weighted_loss(loss, w, min_thres, max_thres, multi=10.0):
    """ heuristic: if loss is low, do not overoptimize, and viceversa """

    # https://www.healthline.com/health/unexplained-weight-loss
    # if loss <= min_thres:
    #     w /= multi  # UNDER-optimize (don't care)

    # if loss >= max_thres:
    #     w *= multi  # OVER-optimize

    return w * loss
