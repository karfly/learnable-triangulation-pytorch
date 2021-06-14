import os
import json
import numpy as np

import torch
from torch.autograd import detect_anomaly

from collections import defaultdict
from itertools import islice

from mvn.utils.misc import live_debug_log
from mvn.utils.vis import save_predictions
from mvn.datasets.utils import prepare_batch
from mvn.pipeline.traditional import batch_iter as original_iter
from mvn.pipeline.dlt_camspace import batch_iter as triangulate_in_cam_iter
from mvn.pipeline.cam2cam import batch_iter as cam2cam_iter


def set_model_state(model, is_train):
    if is_train:
        model.train()
    else:
        model.eval()


def iter_batch(batch, iter_i, model, model_type, criterion, opt, scheduler, config, dataloader, device, epoch, minimon, is_train, cam2cam_model=None, experiment_dir=None):
    indices, cameras, images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = prepare_batch(
        batch, device, config, is_train=is_train
    )
    keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float64)  # 1s, 0s (mainly 1s) ~ 17, 1

    if config.model.cam2cam_estimation:  # predict cam2cam matrices
        results = cam2cam_iter(
            epoch, indices, cameras, iter_i, model, cam2cam_model, opt, scheduler, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon, experiment_dir
        )
    else:  # usual KP estimation
        if config.model.triangulate_in_world_space:  # predict KP in world
            results = original_iter(
                indices, cameras, iter_i, model, model_type, criterion, opt, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, proj_matricies_batch, is_train, config, minimon
            )
        elif config.model.triangulate_in_cam_space:  # predict KP in camspace
            results = triangulate_in_cam_iter(
                indices, cameras, iter_i, model, model_type, criterion, opt, images_batch, keypoints_3d_gt, keypoints_3d_binary_validity_gt, is_train, config, minimon
            )
        else:
            results = None

    if config.debug.write_imgs:  # DC, PD, MP only if necessary: breaks num_workers
        f_out = 'training' if is_train else 'validation'
        f_out += '_batch_{}.png'.format(iter_i)

        batch_size, n_views = images_batch.shape[0], images_batch.shape[1]

        preds = torch.cat([
            torch.cat([
                batch['cameras'][view_i][batch_i].world2proj()(
                    results[batch_i]
                ).unsqueeze(0)
                for view_i in range(n_views)
            ]).unsqueeze(0)  # ~ n_views, 17, 2
            for batch_i in range(batch_size)
        ])  # ~ batch_size, n_views, 17, 2

        gts = torch.cat([
            torch.cat([
                batch['cameras'][view_i][batch_i].world2proj()(
                    keypoints_3d_gt[batch_i]
                ).unsqueeze(0)
                for view_i in range(n_views)
            ]).unsqueeze(0)  # ~ n_views, 17, 2
            for batch_i in range(batch_size)
        ])  # ~ batch_size, n_views, 17, 2

        save_predictions(
            batch,
            images_batch,
            gts.cpu(),
            preds,
            dataloader,
            config,
            batch_out=f_out,
            with_originals=False
        )

    return indices, results


def one_epoch(model, criterion, opt, scheduler, config, dataloader, device, epoch, minimon, is_train=True, master=False, experiment_dir=None, cam2cam_model=None):
    _iter_tag = 'epoch'
    model_type = config.model.name

    set_model_state(model, is_train)

    if config.model.cam2cam_estimation:
        set_model_state(cam2cam_model, is_train)

    results = defaultdict(list)

    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad  # used to turn on/off gradients
    with grad_context():
        iterator = enumerate(dataloader)
        if is_train and config.opt.n_iters_per_epoch is not None:
            iterator = islice(iterator, config.opt.n_iters_per_epoch)

        for iter_i, batch in iterator:  # batch 8 images ~ 384 x 384 => 27.36 KB = 0.0267 MB
            if batch is None:
                print('iter #{:d}: found None batch'.format(iter_i))
                continue

            if config.opt.torch_anomaly_detection:
                torch.autograd.set_detect_anomaly(True)
                with detect_anomaly():  # about x2s time
                    indices_pred, results_pred = iter_batch(
                        batch, iter_i, model, model_type, criterion, opt, scheduler, config, dataloader, device,
                        epoch, minimon, is_train, cam2cam_model=cam2cam_model, experiment_dir=experiment_dir
                    )
            else:
                indices_pred, results_pred = iter_batch(
                    batch, iter_i, model, model_type, criterion, opt, scheduler, config, dataloader, device,
                    epoch, minimon, is_train, cam2cam_model=cam2cam_model, experiment_dir=experiment_dir
                )

            if not (results_pred is None):
                results['preds'].append(results_pred)  # save answers for evaluation
                results['indexes'] += list(indices_pred)

    if master and len(results['preds']) > 0:  # calculate evaluation metrics
        minimon.enter()

        results['preds'] = np.vstack(results['preds'])
        per_pose_error_relative, per_pose_error_absolute, full_metric = dataloader.dataset.evaluate(
            results['preds'],
            indices_predicted=results['indexes'],
            split_by_subject=True
        )  # (average 3D MPJPE (relative to pelvis), all MPJPEs)

        message = '{} MPJPE relative to pelvis: {:.1f} mm, absolute: {:.1f} mm'.format(
            'training' if is_train else 'eval',
            per_pose_error_relative,
            per_pose_error_absolute
        )  # just a little bit of live debug
        live_debug_log(_iter_tag, message)

        minimon.leave('evaluate results')

        if experiment_dir:
            checkpoint_dir = os.path.join(
                experiment_dir, 'checkpoints', '{:04}'.format(epoch)
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            metric_filename = 'metric_train' if is_train else 'metric_eval'
            metric_filename += '.json'
            metric_filename = os.path.join(checkpoint_dir, metric_filename)
            with open(metric_filename, 'w') as fout:
                json.dump(full_metric, fout, indent=4, sort_keys=True)

    if config.debug.show_minimon:
        minimon.print_stats(as_minutes=False)