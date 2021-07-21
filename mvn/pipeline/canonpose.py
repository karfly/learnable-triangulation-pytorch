import os
import numpy as np

import torch

from mvn.models.utils import get_grad_params
from mvn.pipeline.utils import get_kp_gt, backprop
from mvn.pipeline.preprocess import center2pelvis, normalize_keypoints
from mvn.utils.misc import live_debug_log
from mvn.utils.multiview import triangulate_batch_of_points_in_cam_space,homogeneous_to_euclidean, euclidean_to_homogeneous, prepare_cams_for_dlt
from mvn.models.loss import GeodesicLoss, MSESmoothLoss, KeypointsMSESmoothLoss, ProjectionLoss, ScaleIndependentProjectionLoss, BerHuLoss, BodyLoss
from mvn.utils.tred import apply_umeyama, rotate_points

_ITER_TAG = 'canonpose'
PELVIS_I = 6  # H3.6M


def loss_weighted_rep_no_scale(inp, kps_world_pred, cam_rotations_pred, inp_confidences):
    """ the weighted reprojection loss as defined in Equation 5 """

    n_joints = 17  # infer
    n_xy_coords = 2 * n_joints
    n_views = 4  # infer

    def _rotate(kps_world_pred, cam_rotations_pred):
        # reproject to original cameras after applying rotation to the canonical poses
        return torch.cat([
            rotate_points(
                kps_world_pred[i], cam_rotations_pred[i]
            ).unsqueeze(0)
            for i in range(kps_world_pred.shape[0])
        ])

    def _project(points3d):
        return points3d[..., :n_xy_coords]  # only the u,v coordinates are used and depth is ignored (this is a simple weak perspective projection)

    def _flatten(points3d, dims=3):
        return points3d.reshape((-1, n_views, n_joints * dims))

    def _scale(points, dims=2):
        # return flattened_points3d / torch.sqrt(flattened_points3d.square().sum(axis=1, keepdim=True) / n_xy_coords)

        original_shape = points.shape
        points = points.reshape((-1, n_joints * dims))
        return torch.cat([
            (
                points[i] / torch.norm(points[i])  # / n_xy_coords
            ).unsqueeze(0)
            for i in range(points.shape[0])  # each view, across all batches
        ]).reshape(original_shape)

    inp_poses = _project(_flatten(inp, dims=2))  # formally not a projection (these are the 2D detections)
    inp_poses_scaled = _scale(inp_poses)  # normalize by scale

    rot_poses = _project(_flatten(_rotate(kps_world_pred, cam_rotations_pred)))
    rot_poses_scaled = _scale(rot_poses)  # normalize by scale

    diff = (inp_poses_scaled - rot_poses_scaled)\
        .abs()\
        .reshape(-1, 2, n_joints).sum(axis=1)
    conf = (diff * inp_confidences.reshape((-1, n_joints))).sum()
    scale = inp_poses_scaled.shape[0] * inp_poses_scaled.shape[1]
    return conf / scale


def _compute_losses(keypoints_2d, confidences, kps_world_pred, cam_rotations_pred, config):
    dev = kps_world_pred.device
    total_loss = torch.tensor(0.0).to(dev)  # real loss, the one grad is applied to
    loss_weights = config.canonpose.loss

    # reprojection loss
    loss_rep = loss_weighted_rep_no_scale(
        keypoints_2d, kps_world_pred, cam_rotations_pred, confidences
    )
    if loss_weights.rep > 0:
        total_loss += loss_weights.rep * loss_rep

    # view and camera consistency are computed in the same loop
    # "we emphasize that this loss is optional"
    loss_view, loss_camera = torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev)  # todo https://github.com/bastianwandt/CanonPose/blob/main/train.py#L116

    if loss_weights.view > 0:
        total_loss += loss_weights.view * loss_view

    if loss_weights.camera > 0:
        total_loss += loss_weights.camera * loss_camera

    if config.debug.show_live:
        batch_size = keypoints_2d.shape[0]
        __batch_i = np.random.randint(0, batch_size)

        print('pred batch {:.0f}'.format(__batch_i))
        print(kps_world_pred[__batch_i])

    return loss_rep, loss_view, loss_camera, total_loss


def batch_iter(epoch_i, indices, cameras, iter_i, model, opt, scheduler, images_batch, kps_world_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon, experiment_dir):
    def _forward_kp():
        return get_kp_gt(
            kps_world_gt,
            cameras,
            use_extra_cams=0,
            noisy=config.canonpose.data.using_noise
        )

    def _backprop():
        minimon.enter()

        loss_rep, loss_view, loss_camera, total_loss = _compute_losses(
            detections, confidences_pred, kps_world_pred, cam_rotations_pred, config
        )

        minimon.leave('compute loss')

        message = '{} batch iter {:d} losses: SELF REPR ~ {:.3f}, SELF VIEW ~ {:.3f}, SELF CAM ~ {:.3f}, TOTAL ~ {:.0f}'.format(
            'training' if is_train else 'validation',
            iter_i,
            loss_rep.item(),
            loss_view.item(),
            loss_camera.item(),
            total_loss.item()
        )
        live_debug_log(_ITER_TAG, message)

        minimon.enter()
        backprop(
            opt, total_loss, scheduler,
            total_loss, _ITER_TAG, get_grad_params(model)
        )
        minimon.leave('backward pass')

    minimon.enter()
    keypoints_2d_pred, _, confidences_pred = _forward_kp()
    minimon.leave('KPs forward')

    detections = normalize_keypoints(
        keypoints_2d_pred,
        pelvis_center_kps=True,
        normalization=config.ours.preprocess.normalize_kps,
        pelvis_i=PELVIS_I
    ).to('cuda').type(torch.get_default_dtype())
    confidences_pred = confidences_pred.to('cuda').type(torch.get_default_dtype())

    minimon.enter()
    kps_world_pred, cam_rotations_pred = model(
        detections.reshape(-1, 2 * 17),  # flatten along all batches
        confidences_pred.unsqueeze(-1).reshape(-1, 17)
    )

    minimon.leave('forward')

    if is_train:
        _backprop()

    kps_world_pred = kps_world_pred.reshape((-1, 4, 17, 3))
    kps_world_pred = torch.mean(kps_world_pred, axis=1)  # across 1 batch

    if config.canonpose.postprocess.force_pelvis_in_origin:
        kps_world_pred = torch.cat([
            torch.cat([
                kps_world_pred[batch_i] -\
                    kps_world_pred[batch_i, PELVIS_I].unsqueeze(0).repeat(17, 1)
            ]).unsqueeze(0)
            for batch_i in range(kps_world_pred.shape[0])
        ])

    kps_world_pred = apply_umeyama(
        kps_world_gt.to(kps_world_pred.device).type(torch.get_default_dtype()),
        kps_world_pred,
        rotation=False,  # todo check
        scaling=True
    )

    return kps_world_pred.detach().cpu()  # no need for grad no more
