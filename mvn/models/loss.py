from itertools import combinations

import numpy as np

import torch
from torch import nn

from mvn.utils.multiview import euclidean_to_homogeneous, homogeneous_to_euclidean


class KeypointsMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsMSESmoothLoss(nn.Module):
    def __init__(self, threshold=400):
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
    criterion = KeypointsMSESmoothLoss(threshold=3.0)

    for batch_i in range(batch_size):
        cam2cam_gt = torch.cat([
            cam2cam_gts[batch_i][pair[0]][pair[1]].unsqueeze(0)
            for pair in pairs
        ])
        cam2cam_pred = torch.cat([
            cam2cam_preds[batch_i][pair[0]][pair[1]].unsqueeze(0)
            for pair in pairs
        ])

        loss += criterion(
            cam2cam_pred[:, :3, :3].cuda(),  # just R
            cam2cam_gt[:, :3, :3].cuda()
        )  # ~ (len(pairs), )

    return loss


def geo_loss(cam2cam_gts, cam2cam_preds, pairs):
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

        loss += geodesic_distance(
            cam2cam_pred[:, :3, :3].cuda(),  # just R
            cam2cam_gt[:, :3, :3].cuda()
        )  # ~ (len(pairs), )

    return loss


def t_loss(cam2cam_gts, cam2cam_preds, pairs, scale_trans2trans):
    batch_size = cam2cam_gts.shape[0]
    loss = 0.0
    criterion = KeypointsMSESmoothLoss(threshold=400)

    for batch_i in range(batch_size):
        cam2cam_gt = torch.cat([
            cam2cam_gts[batch_i][pair[0]][pair[1]].unsqueeze(0)
            for pair in pairs
        ])
        cam2cam_pred = torch.cat([
            cam2cam_preds[batch_i][pair[0]][pair[1]].unsqueeze(0)
            for pair in pairs
        ])
        
        # todo try use fro norm
        loss += criterion(
            cam2cam_pred[:, :3, 3].cuda() / scale_trans2trans,  # just t
            cam2cam_gt[:, :3, 3].cuda() / scale_trans2trans
        )

    return loss


def tred_loss(keypoints_3d_gt, keypoints_3d_pred, keypoints_3d_binary_validity_gt, scale_keypoints_3d, criterion=KeypointsMSESmoothLoss(threshold=400)):
    return criterion(
        keypoints_3d_pred.cuda() * scale_keypoints_3d,  # ~ 8, 17, 3
        keypoints_3d_gt * scale_keypoints_3d,  # ~ 8, 17, 3
        keypoints_3d_binary_validity_gt.cuda()  # ~ 8, 17, 1
    )


# todo assumption: master cam is 0 => do not project there
def _project_in_each_view(cameras, keypoints_in_cam_pred, cam2cam_preds):
    n_views = len(cameras)
    batch_size = len(cameras[0])

    return torch.cat([
        torch.cat([
            homogeneous_to_euclidean(
                (
                    euclidean_to_homogeneous(keypoints_in_cam_pred[batch_i]).cuda()
                    @
                    cam2cam_preds[batch_i, 0, view_i].T)  # cam 0 -> i
                @
                torch.cuda.FloatTensor(cameras[view_i][batch_i].intrinsics_padded.T)
            ).unsqueeze(0)
            for view_i in range(1, n_views)  # 0 is "master" cam
        ]).unsqueeze(0)  # ~ n_views 1, 3, 17, 2
        for batch_i in range(batch_size)
    ])


def twod_proj_loss(keypoints_3d_gt, cameras, keypoints_in_cam_pred, cam2cam_preds, criterion=KeypointsMSESmoothLoss(threshold=400)):
    n_views = len(cameras)
    batch_size = keypoints_3d_gt.shape[0]

    gts = torch.cat([
        torch.cat([
            cameras[view_i][batch_i].world2proj()(
                keypoints_3d_gt[batch_i]
            ).unsqueeze(0)
            for view_i in range(1, n_views)  # 0 is "master" cam
        ]).unsqueeze(0)  # ~ n_views 3, 17, 2
        for batch_i in range(batch_size)
    ])
    projections = _project_in_each_view(cameras, keypoints_in_cam_pred, cam2cam_preds)

    loss = 0.0

    for batch_i in range(batch_size):
        loss += criterion(
            gts[batch_i].cuda(),
            projections[batch_i].cuda(),
        )

    return loss


def self_consistency_loss(initial_keypoints, cameras, keypoints_in_cam_pred, cam2cam_preds):
    batch_size = cam2cam_preds.shape[0]
    n_views = cam2cam_preds.shape[1]
    pairs = list(combinations(range(n_views), 2))  # on all pairs

    def _self_R(criterion=KeypointsMSESmoothLoss(threshold=3.0)):
        loss = 0.0
        
        for batch_i in range(batch_size):
            cam_i2j = torch.cat([
                cam2cam_preds[batch_i, i, j, :3, :3].unsqueeze(0)
                for i, j in pairs
            ])
            cam_j2i = torch.cat([
                cam2cam_preds[batch_i, j, i, :3, :3].unsqueeze(0)
                for i, j in pairs
            ])

            # cam i -> j should be (cam j -> i) ^ -1 => c_ij * c_ji = I
            pred = torch.bmm(cam_i2j, cam_j2i)

            # comparing VS eye ...
            # ... makes autograd cry
            loss += criterion(  # todo apparently geodesic does not work well ...
                pred,
                torch.eye(3, device=cam2cam_preds.device, requires_grad=True)
            )

            # cam i -> i should be I
            cam_i2i = torch.cat([
                cam2cam_preds[batch_i, i, i, :3, :3].unsqueeze(0)
                for i in range(n_views)
            ])
            loss += criterion(  # todo apparently geodesic does not work well ...
                cam_i2i,
                torch.eye(3, device=cam2cam_preds.device, requires_grad=True)
            )

        return loss

    def _self_t(criterion=KeypointsMSESmoothLoss(threshold=3.0)):
        loss = 0.0

        for batch_i in range(batch_size):  # todo Tensor
            cam_i2j = torch.cat([
                cam2cam_preds[batch_i, i, j].unsqueeze(0)
                for i, j in pairs
            ])
            cam_j2i = torch.cat([
                cam2cam_preds[batch_i, j, i].unsqueeze(0)
                for i, j in pairs
            ])

            # cam i -> j should be (cam j -> i) ^ -1 => c_ij * c_ji = I
            pred = torch.bmm(cam_i2j, cam_j2i)

            loss += criterion(
                pred,
                torch.eye(4, device=cam2cam_preds.device, requires_grad=True)
            )

        return loss

    def _self_projection(criterion=KeypointsMSESmoothLoss(threshold=400)):
        loss = 0.0
        projections = _project_in_each_view(cameras, keypoints_in_cam_pred, cam2cam_preds)  # ~ 8, 4, 17, 2

        for batch_i in range(batch_size):
            loss += criterion(
                initial_keypoints[batch_i, 1:].cuda(),  # not considering master (0)
                projections[batch_i].cuda(),
            )

        return loss

    return _self_R(), _self_t(), _self_projection()
