import os
import shutil
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from mvn.datasets import human36m
from mvn.models.utils import build_opt, show_params, load_checkpoint
from mvn.datasets.utils import worker_init_fn, make_collate_fn
from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.rototrans import RotoTransNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss


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
            crop=config.dataset.train.crop,
            resample_same_K=config.model.cam2cam_estimation,
            look_at_pelvis=config.model.cam2cam_estimation,
            pelvis_in_origin=config.cam2cam.data.pelvis_in_origin,
        )
        print("  training dataset length:", len(train_dataset))

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (train_sampler is None), # debatable
            sampler=train_sampler,
            collate_fn=make_collate_fn(
                randomize_n_views=config.dataset.train.randomize_n_views,
                min_n_views=config.dataset.train.min_n_views,
                max_n_views=config.dataset.train.max_n_views
            ),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=worker_init_fn,
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
        crop=config.dataset.val.crop,
        resample_same_K=config.model.cam2cam_estimation,
        look_at_pelvis=config.model.cam2cam_estimation,
        pelvis_in_origin=config.cam2cam.data.pelvis_in_origin,
    )
    print("  validation dataset length:", len(val_dataset))

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=make_collate_fn(
            randomize_n_views=config.dataset.val.randomize_n_views,
            min_n_views=config.dataset.val.min_n_views,
            max_n_views=config.dataset.val.max_n_views
        ),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, train_sampler


def setup_dataloaders(config, is_train=True, distributed_train=False):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler = setup_human36m_dataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    return train_dataloader, val_dataloader, train_sampler


def setup_experiment(config_path, logdir, config, model_name):
    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = experiment_title

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


def build_env(config, device):
    torch.set_default_dtype(torch.double)  # dio cane

    # build triangulator model ...
    model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet
    }[config.model.name](config, device=device).to(device)

    # ... and cam2cam ...
    if config.model.cam2cam_estimation:
        if config.cam2cam.using_heatmaps:
            roto_net = None  # todo
        else:
            roto_net = RotoTransNet

        cam2cam_model = roto_net(config).to(device)  # todo DistributedDataParallel
    else:
        cam2cam_model = None

    # ... load weights (if necessary) ...

    if config.model.init_weights:
        load_checkpoint(model, config.model.checkpoint)

        print('model:')
        show_params(model, verbose=config.debug.show_models)

    if config.model.cam2cam_estimation:
        if config.cam2cam.model.init_weights:
            load_checkpoint(cam2cam_model, config.cam2cam.model.checkpoint)
        else:
            print('cam2cam model:')
            show_params(cam2cam_model, verbose=config.debug.show_models)

    # ... and opt ...
    opt, scheduler = build_opt(model, cam2cam_model, config)

    # ... and loss criterion
    criterion_class = {
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss
    }[config.opt.criterion]

    if config.opt.criterion == "MSESmooth":
        criterion = criterion_class(config.opt.mse_smooth_threshold)
    else:
        criterion = criterion_class()

    return model, cam2cam_model, criterion, opt, scheduler
