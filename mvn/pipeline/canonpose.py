import os
import numpy as np

import torch

from mvn.models.utils import get_grad_params
from mvn.pipeline.utils import get_kp_gt, backprop
from mvn.pipeline.preprocess import center2pelvis, normalize_keypoints
from mvn.utils.misc import live_debug_log
from mvn.utils.multiview import triangulate_batch_of_points_in_cam_space,homogeneous_to_euclidean, euclidean_to_homogeneous, prepare_cams_for_dlt
from mvn.models.loss import GeodesicLoss, MSESmoothLoss, KeypointsMSESmoothLoss, ProjectionLoss, ScaleIndependentProjectionLoss, BerHuLoss, BodyLoss
from mvn.utils.tred import apply_umeyama

_ITER_TAG = 'canonpose'
PELVIS_I = 6  # H3.6M


def _compute_losses(pred, config):
    dev = pred.device
    total_loss = torch.tensor(0.0).to(dev)  # real loss, the one grad is applied to
    loss_weights = config.canonpose.loss

    # reprojection loss
    loss_rep = torch.tensor(0.0).to(dev)  # todo https://github.com/bastianwandt/CanonPose/blob/main/train.py#L109
    if loss_weights.rep > 0:
        total_loss += loss_weights.rep * loss_rep

    # view and camera consistency are computed in the same loop
    loss_view, loss_camera = torch.tensor(0.0).to(dev), torch.tensor(0.0).to(dev)  # todo https://github.com/bastianwandt/CanonPose/blob/main/train.py#L116

    if loss_weights.view > 0:
        total_loss += loss_weights.view * loss_view

    if loss_weights.camera > 0:
        total_loss += loss_weights.camera * loss_camera

    return loss_rep, loss_view, loss_camera, total_loss


def batch_iter(epoch_i, indices, cameras, iter_i, model, opt, scheduler, images_batch, kps_world_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon, experiment_dir):
    def _forward_kp():  # todo use backbone
        return get_kp_gt(
            kps_world_gt,
            cameras,
            use_extra_cams=0,
            noisy=config.canonpose.data.using_noise
        )

    def _backprop():
        minimon.enter()

        loss_rep, loss_view, loss_camera, total_loss = _compute_losses(
            keypoints_2d_pred,
            config
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
            total_loss,
            _ITER_TAG, get_grad_params(model)
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
    ).to('cuda:0').type(torch.get_default_dtype())  # todo device

    minimon.enter()
    kps_world_pred, cam_angles_pred = model(
        keypoints_2d_pred[0], confidences_pred[0]
    )

    # angles are in axis angle notation -> use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
    cam_rotations_pred = torch.rand(4, 3, 3)  # todo rodrigues(cam_angles_pred)

    # reproject to original cameras after applying rotation to the canonical poses
    rot_poses = cam_rotations_pred.matmul(  # todo use `rotate_points`
        kps_world_pred.reshape(-1, 3, config.canonpose.data.n_joints)
    ).reshape(-1, 3 * config.canonpose.data.n_joints)

    minimon.leave('forward')

    if is_train:
        _backprop()

    return torch.rand(kps_world_gt.shape[0], 17, 3)  # todo kps_world_pred.detach().cpu()  # no need for grad no more
