import os
import shutil
import argparse
import shutil
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice, combinations
import pickle
import copy
import traceback

import numpy as np
import cv2
from PIL import Image

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss, element_weighted_loss, element_by_element
from mvn.models.utils import build_opt

from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.datasets import human36m
from mvn.datasets import utils as dataset_utils
from mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion

from mvn.utils.minimon import MiniMon
from mvn.models.rototrans import Roto6d, compute_geodesic_distance, l2_loss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")

    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--logdir", type=str, required=True, help="Path, where logs will be stored")

    args = parser.parse_args()
    return args


def build_env(config, device):
    model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet
    }[config.model.name](config, device=device).to(device)

    if config.model.cam2cam_estimation:
        cam2cam_model = Roto6d().to(device)  # todo DistributedDataParallel
        opt = optim.Adam(
            [
                {
                    'params': model.backbone.parameters(),
                    'lr': config.opt.lr
                },
                {
                    'params': cam2cam_model.parameters(),
                    'lr': 1e-4
                }
            ],
            lr=config.opt.lr
        )
    else:
        cam2cam_model = None
        opt = build_opt(model, model.backbone.parameters(), config)

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint)
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        print('Successfully loaded pretrained weights for whole model')

    criterion_class = {
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss
    }[config.opt.criterion]

    if config.opt.criterion == "MSESmooth":
        criterion = criterion_class(config.opt.mse_smooth_threshold)
    else:
        criterion = criterion_class()

    return model, cam2cam_model, criterion, opt


def setup_human36m_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        train_dataset = human36m.Human36MMultiViewDataset(
            h36m_root=config.dataset.train.h36m_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.labels_path,
            with_damaged_actions=config.dataset.train.with_damaged_actions,
            retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
            retain_every_n_frames_in_train=config.dataset.train.retain_every_n_frames_in_train,
            scale_bbox=config.dataset.train.scale_bbox,
            kind=config.kind,
            undistort_images=config.dataset.train.undistort_images,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True,
        )
        print("  training dataset length:", len(train_dataset))

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (train_sampler is None), # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(
                randomize_n_views=config.dataset.train.randomize_n_views,
                min_n_views=config.dataset.train.min_n_views,
                max_n_views=config.dataset.train.max_n_views
            ),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    # val
    val_dataset = human36m.Human36MMultiViewDataset(
        h36m_root=config.dataset.val.h36m_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
        labels_path=config.dataset.val.labels_path,
        with_damaged_actions=config.dataset.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        retain_every_n_frames_in_train=config.dataset.train.retain_every_n_frames_in_train,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.dataset.val.undistort_images,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
    )
    print("  validation dataset length:", len(val_dataset))

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(
            randomize_n_views=config.dataset.val.randomize_n_views,
            min_n_views=config.dataset.val.min_n_views,
            max_n_views=config.dataset.val.max_n_views
        ),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, train_sampler


def setup_dataloaders(config, is_train=True, distributed_train=False):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler = setup_human36m_dataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    return train_dataloader, val_dataloader, train_sampler


def setup_experiment(config_path, logdir, config, model_name, is_train=True):
    prefix = "" if is_train else "eval_"

    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = prefix + experiment_title

    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))

    if logdir:
        experiment_dir = os.path.join(logdir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        if config_path:
            shutil.copy(config_path, os.path.join(experiment_dir, "config.yaml"))
    else:
        experiment_dir = None

    return experiment_dir


def original_iter(batch, iter_i, model, model_type, criterion, opt, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, proj_matricies_batch, is_train, config, minimon):
    minimon.enter()

    if model_type == "alg" or model_type == "ransac":
        keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(
            images_batch,
            proj_matricies_batch,  # ~ (batch_size=8, n_views=4, 3, 4)
            minimon
        )  # keypoints_3d_pred, keypoints_2d_pred ~ (8, 17, 3), (~ 8, 4, 17, 2)
    elif model_type == "vol":
        keypoints_3d_pred, heatmaps_pred, volumes_pred, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred = model(
            images_batch,
            proj_matricies_batch,
            batch,
            minimon
        )
    else:  # fail case
        keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None

    minimon.leave('forward pass')

    if is_train:
        use_volumetric_ce_loss = config.opt.use_volumetric_ce_loss if hasattr(config.opt, "use_volumetric_ce_loss") else False

        minimon.enter()

        if config.opt.loss_2d:  # ~ 0 seconds
            batch_size, n_views = images_batch.shape[0], images_batch.shape[1]
            total_loss = 0.0

            for batch_i in range(batch_size):
                for view_i in range(n_views):  # todo faster (batched) loop
                    cam = batch['cameras'][view_i][batch_i]

                    gt = cam.world2proj()(keypoints_3d_gt[batch_i])  # ~ 17, 2
                    pred = cam.world2proj()(
                        keypoints_3d_pred[batch_i]
                    )  # ~ 17, 2

                    total_loss += criterion(
                        pred.unsqueeze(0).cuda(),  # ~ 1, 17, 2
                        gt.unsqueeze(0).cuda(),  # ~ 1, 17, 2
                        keypoints_3d_binary_validity_gt[batch_i].unsqueeze(0).cuda()  # ~ 1, 17, 1
                    )
        elif config.opt.loss_3d:  # ~ 0 seconds
            scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0

            total_loss = criterion(
                keypoints_3d_pred.cuda() * scale_keypoints_3d,  # ~ 8, 17, 3
                keypoints_3d_gt.cuda() * scale_keypoints_3d,  # ~ 8, 17, 3
                keypoints_3d_binary_validity_gt.cuda()  # ~ 8, 17, 1
            )
        elif use_volumetric_ce_loss:
            volumetric_ce_criterion = VolumetricCELoss()

            loss = volumetric_ce_criterion(
                coord_volumes_pred, volumes_pred, keypoints_3d_gt, keypoints_3d_binary_validity_gt
            )

            weight = config.opt.volumetric_ce_loss_weight if hasattr(config.opt, "volumetric_ce_loss_weight") else 1.0

            total_loss += weight * loss

        print('  {} batch iter {:d} loss ~ {:.3f}'.format(
            'training' if is_train else 'validation',
            iter_i,
            total_loss.item()
        ))  # just a little bit of live debug

        minimon.leave('calc loss')

        minimon.enter()

        opt.zero_grad()
        total_loss.backward()  # backward foreach batch

        if hasattr(config.opt, "grad_clip"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.opt.grad_clip / config.opt.lr
            )

        opt.step()

        minimon.leave('backward pass')

    return keypoints_3d_pred.detach().cpu().numpy()


def triangulate_in_cam_iter(batch, iter_i, model, model_type, criterion, opt, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon):
    batch_size, n_views = images_batch.shape[0], images_batch.shape[1]
    master_cams = np.random.randint(0, n_views, size=batch_size)  # choose random "master" cam foreach frame in batch
    proj_matricies_batch = torch.FloatTensor([
        [
            dataset_utils.cam2cam_batch(
                master_cams[batch_i], view_i, batch['cameras'], batch_i
            )
            for view_i in range(n_views)
        ]
        for batch_i in range(batch_size)
    ])

    minimon.enter()

    keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(
        images_batch,
        proj_matricies_batch,  # ~ (batch_size=8, n_views=4, 3, 4)
        minimon
    )

    minimon.leave('BB forward pass')

    if is_train:
        minimon.enter()

        if config.opt.loss_3d:  # variant I: 3D loss on cam KP
            scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0

            gt_in_cam = [
                batch['cameras'][master_cams[batch_i]][batch_i].world2cam()(
                    keypoints_3d_gt[batch_i].cpu()
                ).numpy()
                for batch_i in range(batch_size)
            ]

            total_loss = criterion(
                keypoints_3d_pred.cuda() * scale_keypoints_3d,  # ~ 8, 17, 3
                torch.FloatTensor(gt_in_cam).cuda() * scale_keypoints_3d,  # ~ 8, 17, 3
                keypoints_3d_binary_validity_gt.cuda()  # ~ 8, 17, 1
            )  # "the loss is 3D pose difference between the obtained 3D pose from DLT and the 3D pose in the first camera space"
        else:  # variant II: 2D loss on each view
            total_loss = 0.0

            for batch_i in range(batch_size):
                master_cam = batch['cameras'][master_cams[batch_i]][batch_i]

                for view_i in range(n_views):  # todo faster loop
                    cam = batch['cameras'][view_i][batch_i]

                    gt = cam.world2proj()(keypoints_3d_gt[batch_i].detach().cpu())  # ~ 17, 2
                    pred = master_cam.cam2other(cam)(
                        multiview.euclidean_to_homogeneous(
                            keypoints_3d_pred[batch_i]
                        )
                    )  # ~ 17, 2

                    total_loss += criterion(
                        torch.FloatTensor(pred).unsqueeze(0).cuda(),  # ~ 1, 17, 2
                        torch.FloatTensor(gt).unsqueeze(0).cuda(),
                        keypoints_3d_binary_validity_gt[batch_i].unsqueeze(0).cuda()
                    )  # "The loss is then 2D pose difference between the 2D pose you obtain this way and the GT 2D pose in each view."

        print('  {} batch iter {:d} loss ~ {:.3f}'.format(
            'training' if is_train else 'validation',
            iter_i,
            total_loss.item()
        ))  # just a little bit of live debug

        minimon.leave('calc loss')

        minimon.enter()

        opt.zero_grad()
        total_loss.backward()  # backward foreach batch

        if hasattr(config.opt, "grad_clip"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.opt.grad_clip / config.opt.lr
            )

        opt.step()

        minimon.leave('backward pass')

    # they're in cam space => cam2world for metric evaluation
    keypoints_3d_pred = keypoints_3d_pred.detach().cpu().numpy()
    return np.float32([
        batch['cameras'][master_cams[batch_i]][batch_i].cam2world()(
            keypoints_3d_pred[batch_i]
        )
        for batch_i in range(batch_size)
    ])


def cam2cam_iter(batch, iter_i, model, cam2cam_model, model_type, criterion, opt, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon):
    batch_size, n_views = images_batch.shape[0], images_batch.shape[1]

    minimon.enter()

    keypoints_2d_pred, heatmaps_pred, confidences_pred = model(
        images_batch, None, minimon
    )

    if True:  # using GT KP
        keypoints_2d_pred = torch.zeros(batch_size, n_views, 17, 2)
        for batch_i in range(batch_size):
            for view_i in range(n_views):
                cam = batch['cameras'][view_i][batch_i]
                kp_world = keypoints_3d_gt[batch_i].detach().cpu()  # ~ (17, 3)
                kp_original = cam.world2cam()(kp_world)  # in cam space
                keypoints_2d_pred[batch_i, view_i] = cam.cam2proj()(kp_original)  # ~ (17, 2)
        keypoints_2d_pred.requires_grad = True  # to comply with torch graph

    minimon.leave('BB forward pass')

    minimon.enter()

    n_joints = 17  # todo infer
    pairs = list(sorted(combinations(range(n_views), 2)))  # all sorted combos of cam2cam indices: [(0, 1), (0, 2), ... (2, 3)]
    cam2cam_gts = []
    kps = torch.zeros(batch_size, len(pairs), 2, n_joints, 2)

    for batch_i in range(batch_size):
        cams = [
            batch['cameras'][view_i][batch_i]
            for view_i in range(n_views)
        ]

        kp_in_batch = torch.cat([
            torch.cat([
                keypoints_2d_pred[batch_i, i].unsqueeze(0),  # ~ (1, 17, 2)
                keypoints_2d_pred[batch_i, j].unsqueeze(0)
            ]).unsqueeze(0)
            for (i, j) in pairs
        ])
        kps[batch_i] = kp_in_batch

        # GT roto-translation: (3 x 4 + last row [0, 0, 0, 1]) = [ [ rot2rot | trans2trans ], [0, 0, 0, 1] ]
        cam2cam_gts_batch = [
            cams[j].extrinsics_padded.dot(
                np.linalg.inv(cams[i].extrinsics_padded)
            )  # 4 x 4 roto-translation matrix from cam_i to cam_j
            for (i, j) in pairs
        ]  # ~ (6, 4, 4)

        # GT cam roto-trans from 3D GT world
        # cam2cam_gts_batch = [
        #     cams[view_i].extrinsics  # extrinsics ~ 3 x 4 (R | t)
        #     for view_i in range(n_views)
        # ]

        cam2cam_gts.append(cam2cam_gts_batch)

    cam2cam_gts = torch.FloatTensor(cam2cam_gts).cuda()  # ~ (batch_size=8, len(pairs)=6, 3, 3)
    kps = kps.cuda()

    minimon.leave('prepare GT cam2cam')

    minimon.enter()

    cam2cam_preds = torch.empty(batch_size, len(pairs), 3, 4)

    for batch_i in range(batch_size):
        rot2rot, trans2trans = cam2cam_model(
            kps[batch_i]  # ~ (len(pairs)=6, 2, n_joints=17, 2D)
        )
        trans2trans = trans2trans.unsqueeze(0).view(len(pairs), 3, 1)  # .T
        cam2cam_preds[batch_i] = torch.cat([
            rot2rot, trans2trans
        ], dim=2)  # `torch.hstack`, for compatibility with cluster

    minimon.leave('cam2cam forward pass')

    minimon.enter()

    full_cam2cam = torch.empty(batch_size, len(pairs), 4, 4, device='cuda:0')
    for batch_i in range(batch_size):
        for pair_i in range(len(pairs)):
            R = cam2cam_preds[batch_i, pair_i, :3, :3].cuda()
            t = cam2cam_preds[batch_i, pair_i, :3, 3].cuda()

            cam2cam = torch.cat([R, t.unsqueeze(0).T], dim=1)  # `torch.hstack`, for compatibility with cluster
            cam2cam = torch.cat([  # `torch.vstack`, for compatibility with cluster
                cam2cam,
                torch.cuda.FloatTensor([0, 0, 0, 1]).unsqueeze(0)
            ], dim=0)  # 4 x 4
            full_cam2cam[batch_i, pair_i] = cam2cam

    # now that I have all cam2cam ...
    master_cams = [0] * batch_size  # todo random
    full_cam2cam = [
        [None] + list(full_cam2cam[batch_i][0: 2 + 1])
        for batch_i in range(batch_size)
    ]  # I only need [(0, 0)=None, (0, 1), (0, 2), (0, 3)] since 0 is the master
    full_cam2cam = torch.cat([
        torch.cat([
            dataset_utils.cam2cam_precomputed_batch(
                master_cams[batch_i], view_i, batch['cameras'], batch_i, full_cam2cam
            ).unsqueeze(0)
            for view_i in range(n_views)
        ]).unsqueeze(0)
        for batch_i in range(batch_size)
    ])

    # ... DLT in one random master cam but since ...
    keypoints_3d_pred = multiview.triangulate_batch_of_points_in_cam_space(
        full_cam2cam.cpu(),
        keypoints_2d_pred.cpu(),
        confidences_batch=confidences_pred.cpu()
    )

    # ... they're in master cam space => cam2world
    keypoints_3d_pred = torch.cat([
        batch['cameras'][master_cams[batch_i]][batch_i].cam2world()(
            keypoints_3d_pred[batch_i]
        ).unsqueeze(0)
        for batch_i in range(batch_size)
    ])

    minimon.leave('cam2cam DLT')

    if is_train:
        total_loss = 0.0

        minimon.enter()

        for batch_i in range(batch_size):
            # geodesic loss
            loss = compute_geodesic_distance()(
                cam2cam_preds[batch_i, :, :3, :3].cuda(),
                cam2cam_gts[batch_i, :, :3, :3].cuda()
            )  # ~ (len(pairs), )
            total_loss += loss

            # trans loss
            weight_trans = 0.1
            if weight_trans > 0:
                loss = l2_loss()(
                    cam2cam_preds[batch_i, :, :3, 3].cuda(),
                    cam2cam_gts[batch_i, :, :3, 3].cuda()
                )
                total_loss += weight_trans * loss

            # 2D KP loss
            weight_kp = 0.0  # todo not predicting them (for now)
            if weight_kp > 0:
                for view_i in range(n_views):
                    cam = batch['cameras'][view_i][batch_i]

                    gt = cam.world2proj()(keypoints_3d_gt[batch_i])  # ~ 17, 2
                    pred = cam.world2proj()(  # todo faster (batched) loop
                        keypoints_3d_pred[batch_i]
                    )  # ~ 17, 2

                    total_loss += weight_kp * criterion(
                        pred.unsqueeze(0).cuda(),  # ~ 1, 17, 2
                        gt.unsqueeze(0).cuda(),  # ~ 1, 17, 2
                        keypoints_3d_binary_validity_gt[batch_i].unsqueeze(0).cuda()  # ~ 1, 17, 1
                    )

            # pose loss
            weight_pose = 0.0
            if weight_pose > 0:
                in_world = torch.cuda.FloatTensor(keypoints_3d_gt[batch_i])
                for view_i in range(n_views):
                    cam = batch['cameras'][view_i][batch_i]
                    gt = cam.world2proj()(in_world)  # ~ 17, 2

                    pred = cam.world2proj()(keypoints_3d_pred[batch_i])  # ~ 17, 2

                    total_loss += weight_pose * criterion(
                        pred.unsqueeze(0).cuda(),  # ~ 1, 17, 2
                        gt.unsqueeze(0).cuda(),  # ~ 1, 17, 2
                        keypoints_3d_binary_validity_gt[batch_i].unsqueeze(0).cuda()  # ~ 1, 17, 1
                    )

        minimon.leave('calc loss')

        print('  {} batch iter {:d} loss ~ {:.3f}'.format(
            'training' if is_train else 'validation',
            iter_i,
            total_loss.item()
        ))  # just a little bit of live debug

        minimon.enter()

        opt.zero_grad()
        total_loss.backward()  # backward foreach batch

        if hasattr(config.opt, "grad_clip"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.opt.grad_clip / config.opt.lr
            )

        opt.step()

        minimon.leave('backward pass')

    return keypoints_3d_pred.detach()


def iter_batch(batch, iter_i, model, model_type, criterion, opt, config, dataloader, device, epoch, minimon, is_train, cam2cam_model=None):
    images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(
        batch, device, config
    )
    keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)  # 1s, 0s (mainly 1s) ~ 17, 1

    if config.model.cam2cam_estimation:  # predict cam2cam matrices
        results = cam2cam_iter(
            batch, iter_i, model, cam2cam_model, model_type, criterion, opt, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon
        )
    else:  # usual KP estimation
        if config.model.triangulate_in_world_space:  # predict KP in world
            results = original_iter(
                batch, iter_i, model, model_type, criterion, opt, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, proj_matricies_batch, is_train, config, minimon
            )
        elif config.model.triangulate_in_cam_space:  # predict KP in camspace
            results = triangulate_in_cam_iter(
                batch, iter_i, model, model_type, criterion, opt, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon
            )
        else:
            results = None

    return results


def set_model_state(model, is_train):
    if is_train:
        model.train()
    else:
        model.eval()


def one_epoch(model, criterion, opt, config, dataloader, device, epoch, minimon, is_train=True, master=False, experiment_dir=None, cam2cam_model=None):
    model_type = config.model.name

    set_model_state(model, is_train)

    if config.model.cam2cam_estimation:  # also using `cam2cam_model`
        set_model_state(cam2cam_model, is_train)

    results = defaultdict(list)

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        iterator = enumerate(dataloader)
        if is_train and config.opt.n_iters_per_epoch is not None:
            iterator = islice(iterator, config.opt.n_iters_per_epoch)

        for iter_i, batch in iterator:  # batch 8 images ~ 384 x 384 => 27.36 KB = 0.0267 MB
            if batch is None:
                print('iter #{:d}: found None batch'.format(iter_i))
                continue

            if config.opt.torch_anomaly_detection:
                with autograd.detect_anomaly():  # about x2s time
                    results_pred = iter_batch(
                        batch, iter_i, model, model_type, criterion, opt, config, dataloader, device,
                        epoch, minimon, is_train, cam2cam_model=cam2cam_model
                    )
            else:
                results_pred = iter_batch(
                    batch, iter_i, model, model_type, criterion, opt, config, dataloader, device,
                    epoch, minimon, is_train, cam2cam_model=cam2cam_model
                )

            if not (results_pred is None):
                results['preds'].append(results_pred)  # save answers for evaluation
                results['indexes'] += batch['indexes']

    if master and len(results['preds']) > 0:  # calculate evaluation metrics
        minimon.enter()

        results['preds'] = np.vstack(results['preds'])
        scalar_metric, full_metric = dataloader.dataset.evaluate(
            results['preds'],
            indices_predicted=results['indexes']
        )  # (average 3D MPJPE (relative to pelvis), all MPJPEs)

        print('  {} MPJPE relative to pelvis: {:.3f} mm'.format(
            'training' if is_train else 'eval',
            scalar_metric
        ))  # just a little bit of live debug

        minimon.leave('evaluate results')

        checkpoint_dir = os.path.join(
            experiment_dir, 'checkpoints', '{:04}'.format(epoch)
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        metric_filename = 'metric_train' if is_train else 'metric_eval'
        metric_filename += '.json'
        metric_filename = os.path.join(checkpoint_dir, metric_filename)
        with open(metric_filename, 'w') as fout:
            json.dump(full_metric, fout, indent=4, sort_keys=True)

        if config.debug.dump_results and experiment_dir and not is_train:
            results_filename = os.path.join(checkpoint_dir, 'results.pkl')
            with open(results_filename, 'wb') as fout:
                pickle.dump(results, fout)


def init_distributed(args):
    if not misc.is_distributed():
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True


def do_train(config_path, logdir, config, device, is_distributed, master):
    model, cam2cam_model, criterion, opt = build_env(config, device)
    if is_distributed:  # multi-gpu
        model = DistributedDataParallel(model, device_ids=[device])

    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)  # ~ 0 seconds

    if master:
        experiment_dir = setup_experiment(
            config_path, logdir, config, type(model).__name__, is_train=True
        )
    else:
        experiment_dir = None

    minimon = MiniMon()

    for epoch in range(config.opt.n_epochs):  # training
        if master:
            f_out = 'epoch {:4d} has started!'
            print(f_out.format(epoch))

        if train_sampler:  # None when NOT distributed
            train_sampler.set_epoch(epoch)

        minimon.enter()
        one_epoch(
            model, criterion, opt, config, train_dataloader, device, epoch,
            minimon, is_train=True, master=master, experiment_dir=experiment_dir, cam2cam_model=cam2cam_model
        )
        minimon.leave('do train')

        minimon.enter()
        one_epoch(
            model, criterion, opt, config, val_dataloader, device, epoch,
            minimon, is_train=False, master=master, experiment_dir=experiment_dir, cam2cam_model=cam2cam_model
        )
        minimon.leave('do eval')

        if master and experiment_dir and config.debug.dump_checkpoints:
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)

            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights_model.pth"))

            if config.model.cam2cam_estimation:
                torch.save(cam2cam_model.state_dict(), os.path.join(checkpoint_dir, "weights_cam2cam_model.pth"))

            f_out = 'epoch {:4d} complete!'
            print(f_out.format(epoch))

            minimon.print_stats(as_minutes=False)
            print('=' * 105)

    if master:
        minimon.print_stats(as_minutes=False)


def main(args):
    print('# available GPUs: {:d}'.format(torch.cuda.device_count()))

    is_distributed = init_distributed(args)
    master = misc.is_master()
    device = torch.device(args.local_rank) if is_distributed else torch.device(0)
    print('using dev {}'.format(device))

    config = cfg.load_config(args.config)
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    do_train(args.config, args.logdir, config, device, is_distributed, master)


if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))

    main(args)
