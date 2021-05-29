from itertools import combinations

import numpy as np

import torch
from torch import nn
from torch import tensor

from mvn.utils.multiview import _2proj, _my_proj
from mvn.utils.misc import get_pairs, get_master_pairs
from mvn.utils.img import rotation_matrix_from_vectors_torch


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


# todo as module
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


class HuberLoss(nn.Module):
    def __init__(self, threshold):
        super().__init__()

        self.c = np.float64(threshold)

    def _criterion(self, diff):
        diff[torch.abs(diff) <= self.c] = torch.abs(diff[torch.abs(diff) <= self.c])  # L1 norm within threshold
        
        diff[torch.abs(diff) > self.c] =\
            (torch.square(diff[torch.abs(diff) > self.c]) + np.square(self.c)) / (2 * self.c)

        return diff.mean()

    def forward(self, pred, gt):
        diff = pred - gt
        return self._criterion(diff)


class KeypointsRotoLoss(nn.Module):
    def __init__(self, threshold=20*20, w_R=1e1, w_t=1e-1):
        super().__init__()

        self.threshold = threshold  # todo use it
        self.w_R = w_R
        self.w_t = w_t

    def forward(self, pred, gt, valids=None):
        batch_size = pred.shape[0]
        n_joints = pred.shape[1]
        dev = 'cuda'
        
        loss_R = torch.tensor(0.0).to(dev)
        loss_t = torch.tensor(0.0).to(dev)

        for batch_i in range(batch_size):  # todo tensored
            for joint_i in range(n_joints):
                is_origin = torch.norm(pred[batch_i, joint_i] - torch.zeros(3)) < 1e-3 or torch.norm(gt[batch_i, joint_i] - torch.zeros(3)) < 1e-3
                if not is_origin:
                    loss_t += torch.norm(pred[batch_i, joint_i] - gt[batch_i, joint_i])

                    Rt = rotation_matrix_from_vectors_torch(
                        pred[batch_i, joint_i],
                        gt[batch_i, joint_i]
                    )
                    loss_R += geodesic_distance(
                        Rt.unsqueeze(0).to(dev),
                        torch.eye(3).unsqueeze(0).to(dev)
                    )
        
        total_loss = self.w_R * loss_R + self.w_t * loss_t
        normalization = batch_size  # * n_joints
        return total_loss / normalization


def geo_loss(gts, preds, criterion=geodesic_distance):
    n_cameras = gts.shape[1]
    batch_size = gts.shape[0]
    dev = preds.device

    return criterion(
        preds.view(batch_size * n_cameras, 4, 4)[:, :3, :3],  # just R
        gts.view(batch_size * n_cameras, 4, 4)[:, :3, :3].to(dev)
    )


def t_loss(gts, preds, scale_t, criterion=MSESmoothLoss(threshold=4e2)):
    dev = preds.device
    n_cameras = gts.shape[1]
    batch_size = gts.shape[0]
    return criterion(
        preds.view(batch_size * n_cameras, 4, 4)[:, :3, 3].to(dev) / scale_t,  # just t
        gts.view(batch_size * n_cameras, 4, 4)[:, :3, 3].to(dev) / scale_t
    )


def tred_loss(preds, gts, keypoints_3d_binary_validity_gt, scale_keypoints_3d, criterion=KeypointsRotoLoss(threshold=20*20)):
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
                for view_i in range(1, n_views)
            ]).to(dev),  # gt
        ).unsqueeze(0)
        for batch_i in range(batch_size)
    ]))


def _self_consistency_cam(cams_preds, scale_t):
    ordered_views = get_master_pairs()
    n_cams = cams_preds.shape[0]
    batch_size = cams_preds.shape[1]
    dev = cams_preds.device

    loss_R = torch.tensor(0.0).to(dev)
    loss_t = torch.tensor(0.0).to(dev)

    comparisons = list(combinations(range(n_cams), 2))  # pair comparison
    for cam_i in range(1):  # todo tensored
        index_cam = [
            master_views.index(cam_i)
            for master_views in ordered_views
        ]
        for batch_i in range(batch_size):  # todo tensored
            cams = torch.cat([
                cams_preds[row_i, batch_i, i].unsqueeze(0)
                for row_i, i in enumerate(index_cam)
            ])  # same extrinsics cam but predicted from a different master ...

            compare_i = torch.cat([
                cams[:, :3, :3][i].unsqueeze(0)  # just R
                for i, _ in comparisons
            ])
            compare_j = torch.cat([
                cams[:, :3, :3][j].unsqueeze(0)
                for _, j in comparisons
            ])  # todo batched
            loss_R += geodesic_distance(compare_i, compare_j)
            
            norm_t = torch.square(torch.mean(torch.cat([
                torch.norm(cams[:, 2, 3][i]).unsqueeze(0)
                for i in range(n_cams)
            ])))
            compare_i = torch.cat([
                cams[:, 2, 3][i].unsqueeze(0) / scale_t  # just t
                for i, _ in comparisons
            ])
            compare_j = torch.cat([
                cams[:, 2, 3][j].unsqueeze(0) / scale_t
                for _, j in comparisons
            ])
            loss_t += MSESmoothLoss(threshold=1e1)(compare_i, compare_j) * norm_t  # penalize large vectors

    normalization = n_cams * batch_size
    loss_R = loss_R / normalization * 1e1
    loss_t = loss_t / normalization

    return loss_R + loss_t


def _self_consistency_world(kps_world_pred, scale_keypoints_3d):
    n_cams = kps_world_pred.shape[0]
    dev = kps_world_pred.device
    loss = torch.tensor(0.0).to(dev)

    for master_i in range(n_cams):  # todo tensored
        kps_world_predicted_as_master = kps_world_pred[master_i]
        kps_predicted_by_the_others = torch.cat([
            kps_world_pred[other_master_i].unsqueeze(0)
            for other_master_i in range(n_cams)
            if other_master_i != master_i
        ]).mean(axis=0)
        loss += KeypointsMSESmoothLoss(threshold=20*20)(
            kps_world_predicted_as_master.unsqueeze(0),
            kps_predicted_by_the_others.unsqueeze(0),
        ) / torch.sqrt(torch.norm(kps_world_predicted_as_master, p='fro'))  # penalize trivials

    normalization = n_cams
    return loss / normalization


def _self_consistency_2D(same_K_for_all, cam_preds, kps_world_pred, initial_keypoints, criterion=KeypointsMSESmoothLoss(threshold=1e2), scale_kps=1e2):
    n_views = cam_preds.shape[0]
    batch_size = cam_preds.shape[1]
    dev = cam_preds.device
    pairs = get_pairs()
    loss = torch.tensor(0.0).to(dev)

    for master_cam_i in range(n_views):
        projections = torch.cat([
            torch.cat([
                _my_proj(
                    cam_preds[master_cam_i, batch_i, view_i],
                    same_K_for_all.to(dev)
                )(kps_world_pred[batch_i]).unsqueeze(0)
                for view_i in range(1, n_views)  # not considering master (0)
            ]).unsqueeze(0).to(dev)  # pred
            for batch_i in range(batch_size)
        ])

        loss += torch.mean(
            torch.cat([
                criterion(
                    torch.cat([
                        initial_keypoints[batch_i, i].unsqueeze(0) * scale_kps
                        for _, i in pairs[master_cam_i]
                    ]).to(dev),  # gt
                    projections[batch_i] * scale_kps
                ).unsqueeze(0) / torch.sqrt(torch.norm(projections[batch_i], p='fro'))  # penalize trivials
                for batch_i in range(batch_size)
            ])
        )

    normalization = n_views
    return loss / normalization


def _self_separation(keypoints_cam_pred):
    """ see eq 8 in https://papers.nips.cc/paper/2018/file/24146db4eb48c718b84cae0a0799dcfc-Paper.pdf """

    return None


def self_consistency_loss(cameras, cam_preds, kps_world_pred, initial_keypoints, master_cam_i, scale_t, scale_keypoints_3d):
    loss_cam2cam = _self_consistency_cam(cam_preds, scale_t)
    loss_proj = _self_consistency_2D(
        torch.DoubleTensor(cameras[0][0].intrinsics_padded),
        cam_preds,
        kps_world_pred.mean(axis=0),
        initial_keypoints
    )
    loss_world = _self_consistency_world(kps_world_pred, scale_keypoints_3d)
    # todo loss_sep = _self_separation(keypoints_cam_pred)
    return loss_cam2cam, loss_proj, loss_world  # todo and others


def get_weighted_loss(loss, w, min_thres, max_thres, multi=10.0):
    """ heuristic: if loss is low, do not overoptimize, and viceversa """

    # https://www.healthline.com/health/unexplained-weight-loss
    # if loss <= min_thres:
    #     w /= multi  # UNDER-optimize (don't care)

    # if loss >= max_thres:
    #     w *= multi  # OVER-optimize

    return w * loss
