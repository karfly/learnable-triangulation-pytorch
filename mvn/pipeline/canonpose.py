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

_ITER_TAG = 'cam2cam'
PELVIS_I = 6  # H3.6M


def batch_iter(epoch_i, indices, cameras, iter_i, model, opt, scheduler, images_batch, kps_world_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon, experiment_dir):
    1/0