import numpy as np

import torch
from torch import nn

from mvn.utils.multiview import homogeneous_to_euclidean, euclidean_to_homogeneous
from mvn.utils.misc import get_pairs, get_inverse_i_from_pair
from mvn.models.layers import RodriguesBlock


class KeypointsMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsMSESmoothLoss(nn.Module):
    def __init__(self, threshold=20*20):
        super().__init__()

        self.threshold = threshold

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity=None):
        dimension = keypoints_pred.shape[-1]
        diff = (keypoints_gt - keypoints_pred) ** 2

        if not (keypoints_binary_validity is None):
            diff *= keypoints_binary_validity

        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)  # soft version
        loss = torch.sum(diff) / dimension
        
        if not (keypoints_binary_validity is None):
            loss /= max(1, torch.sum(keypoints_binary_validity).item())

        return loss


class MSESmoothLoss(nn.Module):
    def __init__(self, threshold):
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


def L2_R_loss(cam2cam_gts, cam2cam_preds, pairs):
    batch_size = cam2cam_gts.shape[0]
    loss = 0.0
    
    for batch_i in range(batch_size):
        cam2cam_gt = torch.cat([
            cam2cam_gts[batch_i][pair[0]][pair[1]].unsqueeze(0)
            for pair in pairs
        ])
        cam2cam_pred = torch.cat([
            cam2cam_preds[batch_i][pair[0]][pair[1]].unsqueeze(0)
            for pair in pairs
        ])

        loss += KeypointsMSESmoothLoss(threshold=0.5)(
            cam2cam_pred[:, :3, :3].cuda(),  # just R
            cam2cam_gt[:, :3, :3].cuda()
        )  # ~ (len(pairs), )

    return loss / batch_size


def geo_loss(cam2cam_gts, cam2cam_preds, criterion=geodesic_distance):
    n_cameras = cam2cam_gts.shape[0]
    n_pairs = n_cameras - 1
    batch_size = cam2cam_gts.shape[1]

    loss = torch.tensor(0.0).to('cuda')
    for master_cam_i in range(n_cameras):
        loss += criterion(
            cam2cam_preds[master_cam_i].view(
                batch_size * n_pairs, 4, 4
            )[:, :3, :3].cuda(),  # just R
            cam2cam_gts[master_cam_i].view(
                batch_size * n_pairs, 4, 4
            )[:, :3, :3].cuda()
        )

    return loss / n_cameras


def t_loss(cam2cam_gts, cam2cam_preds, scale_trans2trans, criterion=MSESmoothLoss(threshold=1e2)):
    n_cameras = cam2cam_gts.shape[0]
    n_pairs = n_cameras - 1
    batch_size = cam2cam_gts.shape[1]

    loss = torch.tensor(0.0).to('cuda')
    for master_cam_i in range(n_cameras):
        loss += criterion(
            cam2cam_preds[master_cam_i].view(
                batch_size * n_pairs, 4, 4
            )[:, :3, 3].cuda() / scale_trans2trans,  # just t
            cam2cam_gts[master_cam_i].view(
                batch_size * n_pairs, 4, 4
            )[:, :3, 3].cuda() / scale_trans2trans
        )

    return loss / n_cameras


def tred_loss(keypoints_3d_gt, keypoints_3d_pred, keypoints_3d_binary_validity_gt, scale_keypoints_3d, criterion=KeypointsMSESmoothLoss(threshold=20*20)):
    return criterion(
        keypoints_3d_pred.cuda() * scale_keypoints_3d,  # ~ 8, 17, 3
        keypoints_3d_gt * scale_keypoints_3d,  # ~ 8, 17, 3
        keypoints_3d_binary_validity_gt.cuda()  # ~ 8, 17, 1
    ).to(keypoints_3d_pred.device)


def twod_proj_loss(keypoints_3d_gt, keypoints_3d_pred, cameras, criterion=KeypointsMSESmoothLoss(threshold=10*10)):
    """ project GT VS pred points to all views """

    n_views = len(cameras)
    batch_size = keypoints_3d_gt.shape[0]
    loss = 0.0
    
    for batch_i in range(batch_size):
        gt = torch.cat([
            cameras[view_i][batch_i].world2proj()(
                keypoints_3d_gt[batch_i]
            ).unsqueeze(0)
            for view_i in range(1, n_views)  # 0 is "master" cam
        ])  # ~ n_views - 1, 17, 2
        
        pred = torch.cat([
            cameras[view_i][batch_i].world2proj()(
                keypoints_3d_pred[batch_i]
            ).unsqueeze(0)
            for view_i in range(1, n_views)  # not considering master (0)
        ])  # ~ n_views - 1, 17, 2
    
        loss += criterion(
            gt.cuda(),
            pred.cuda(),
        )

    return loss / batch_size


def _project_in_other_views(cameras, keypoints_mastercam_pred, cam2cam_preds, master_cam_i):
    batch_size = len(cameras[0])
    pairs = get_pairs()[master_cam_i]

    return torch.cat([
        torch.cat([
            homogeneous_to_euclidean(
                (
                    euclidean_to_homogeneous(keypoints_mastercam_pred[batch_i]).cuda()
                    @
                    cam2cam_preds[master_cam_i, batch_i, i].T)  # cam master -> i
                @
                torch.cuda.DoubleTensor(cameras[target][batch_i].intrinsics_padded.T)
            ).unsqueeze(0)
            for i, (_, target) in enumerate(pairs)
        ]).unsqueeze(0)  # ~ n_views 1, 3, 17, 2
        for batch_i in range(batch_size)
    ])


def _self_consistency_ext(cam2cam_preds, keypoints_cam_pred, master_cam_i=0, criterion=KeypointsMSESmoothLoss(threshold=20*20), pelvis_i=6):
    """ project mastercam KPs to another cam, and reproject back """

    batch_size = keypoints_cam_pred.shape[0]
    n_comparisons = cam2cam_preds.shape[2]
    pairs = get_pairs()[master_cam_i]
    inverses = torch.cat([
        cam2cam_preds[other_cam, :, get_inverse_i_from_pair(master_cam_i, other_cam)[1]].unsqueeze(0)
        for _, other_cam in pairs
    ])

    loss_on_kps = torch.tensor(0.0).to(keypoints_cam_pred.device)

    for batch_i in range(batch_size):  # todo tensored
        kp_in_mastercam = keypoints_cam_pred[batch_i]
        pelvis_mastercam_d = kp_in_mastercam[pelvis_i]
        kp_in_mastercam_pelvis_centered = kp_in_mastercam - pelvis_mastercam_d

        invs = inverses[:, batch_i]

        for other_view in range(n_comparisons):
            to_and_from_mat = torch.mm(
                cam2cam_preds[master_cam_i, batch_i, other_view],
                invs[other_view]
            ).to(keypoints_cam_pred.device)
            preds = homogeneous_to_euclidean(
                torch.mm(
                    euclidean_to_homogeneous(kp_in_mastercam),
                    to_and_from_mat
                )
            )
            pelvis_d = preds[pelvis_i]
            preds_pelvis_centered = preds - pelvis_d

            # loss_on_pelvis_distance += MSESmoothLoss(threshold=1e3)(
            #     pelvis_mastercam_d.unsqueeze(0),
            #     pelvis_d.unsqueeze(0),
            # )

            norms = torch.norm(kp_in_mastercam_pelvis_centered, p='fro') +\
                torch.norm(preds_pelvis_centered, p='fro')
            loss_on_kps += criterion(
                kp_in_mastercam_pelvis_centered.unsqueeze(0),
                preds_pelvis_centered.unsqueeze(0)
            ) / norms

    normalization = n_comparisons * batch_size
    return loss_on_kps / normalization

    # loss_R, loss_t = torch.tensor(0.0).to('cuda'), torch.tensor(0.0).to('cuda')
    # for master_cam_i in range(n_cameras):
    #     pairs = get_pairs()[master_cam_i]
    #     inverses = torch.cat([
    #         cam2cam_preds[other_cam, :, get_inverse_i_from_pair(master_cam_i, other_cam)[1]].unsqueeze(0)
    #         for _, other_cam in pairs
    #     ])

    #     for batch_i in range(batch_size):
    #         # rotation
    #         pred = torch.bmm(
    #             cam2cam_preds[master_cam_i, batch_i],
    #             inverses[:, batch_i]
    #         )[:, :3, :3]
    #         gt = torch.eye(  # comparing VS eye ... makes autograd cry
    #             3, device=cam2cam_preds.device, requires_grad=False
    #         ).unsqueeze(0).repeat((n_pairs, 1, 1))
    #         loss_R += geodesic_distance(pred, gt)  # todo add rot i -> i

    #         # translation
    #         pred = cam2cam_preds[master_cam_i, batch_i, :, :3, 3]
    #         gt = torch.inverse(inverses[:, batch_i])[:, :3, 3]
    #         loss_t += MSESmoothLoss(threshold=1e3)(pred, gt) / np.sqrt(scale_trans2trans)

    # w_R, w_t = 1.0, 1.0
    # normalization = batch_size * n_cameras
    # return w_R * loss_R / normalization +\
    #     w_t * loss_t / normalization


def _self_consistency_P(cameras, cam2cam_preds, keypoints_cam_pred, initial_keypoints, master_cam_i, criterion=KeypointsMSESmoothLoss(threshold=10*10)):
    projections = _project_in_other_views(
        cameras, keypoints_cam_pred, cam2cam_preds, master_cam_i
    )  # ~ 8, 3, 17, 2

    batch_size = len(cameras[0])
    pairs = get_pairs()[master_cam_i]

    loss = torch.tensor(0.0).to(keypoints_cam_pred.device)
    for batch_i in range(batch_size):
        kps = torch.cat([
            initial_keypoints[batch_i, i].unsqueeze(0)
            for _, i in pairs
        ])

        norms = torch.norm(kps, p='fro') +\
            torch.norm(projections[batch_i], p='fro')
        loss += criterion(
            kps.to(keypoints_cam_pred.device),
            projections[batch_i].to(keypoints_cam_pred.device),
        ) / norms.to(keypoints_cam_pred.device)

    return loss


def self_consistency_loss(cameras, cam2cam_preds, scale_trans2trans, keypoints_cam_pred, initial_keypoints, master_cam_i):

    loss_ext = _self_consistency_ext(cam2cam_preds, keypoints_cam_pred)
    loss_proj = _self_consistency_P(cameras, cam2cam_preds, keypoints_cam_pred, initial_keypoints, master_cam_i)

    return loss_ext, loss_proj


def get_weighted_loss(loss, w, min_thres, max_thres, multi=10.0):
    """ heuristic: if loss is low, do not overoptimize, and viceversa """

    # https://www.healthline.com/health/unexplained-weight-loss
    # if loss <= min_thres:
    #     w /= multi  # UNDER-optimize (don't care)

    # if loss >= max_thres:
    #     w *= multi  # OVER-optimize

    return w * loss
