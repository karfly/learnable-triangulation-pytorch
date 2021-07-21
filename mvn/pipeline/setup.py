import os
import shutil
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from mvn.datasets import human36m
from mvn.models.utils import show_params, load_checkpoint, get_grad_params, freeze_backbone
from mvn.datasets.utils import worker_init_fn, make_collate_fn
from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.rototrans import RotoTransNet, Cam2camNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss
from mvn.models.canonpose import CanonPose


def setup_human36m_dataloaders(config, is_train, distributed_train):
    resample_same_K = True
    look_at_pelvis = config.pipeline.model == 'ours' and config.ours.triangulate.master and config.ours.data.look_at_pelvis
    pelvis_in_origin = config.ours.data.pelvis_in_origin and config.ours.cams.project == 'pinhole'
    scale2meters = config.ours.preprocess.scale2meters
    image_shape = config.image_shape if hasattr(config, "image_shape") else (256, 256)
    scale_bbox = config.dataset.train.scale_bbox,
    kind = config.kind
    opt = config[config.pipeline.model].opt

    if is_train:
        train_dataset = human36m.Human36MMultiViewDataset(
            h36m_root=config.dataset.train.h36m_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=image_shape,
            labels_path=config.dataset.train.labels_path,
            with_damaged_actions=config.dataset.train.with_damaged_actions,
            retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
            retain_every_n_frames_in_train=config.dataset.train.retain_every_n_frames_in_train,
            scale_bbox=scale_bbox,
            kind=kind,
            undistort_images=config.dataset.train.undistort_images,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop,
            resample_same_K=resample_same_K,
            look_at_pelvis=look_at_pelvis,
            pelvis_in_origin=pelvis_in_origin,
            scale2meters=scale2meters
        )
        print("  training dataset length:", len(train_dataset))

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
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
    else:
        train_dataloader = None

    # val
    val_dataset = human36m.Human36MMultiViewDataset(
        h36m_root=config.dataset.val.h36m_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=image_shape,
        labels_path=config.dataset.val.labels_path,
        with_damaged_actions=config.dataset.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        retain_every_n_frames_in_train=config.dataset.train.retain_every_n_frames_in_train,
        scale_bbox=scale_bbox,
        kind=kind,
        undistort_images=config.dataset.val.undistort_images,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop,
        resample_same_K=resample_same_K,
        look_at_pelvis=look_at_pelvis,
        pelvis_in_origin=pelvis_in_origin,
        scale2meters=scale2meters
    )
    print("  validation dataset length:", len(val_dataset))

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opt.val_batch_size,
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

    base_optim = optim.Adam  # if _get_torch_version() >= 1.8 else optim.AdamW

    if config.pipeline.model == 'classic':
        model = {
            "ransac": RANSACTriangulationNet,
            "alg": AlgebraicTriangulationNet,
            "vol": VolumetricTriangulationNet
        }[config.model.name](config, device=device).to(device)

        if config.model.init_weights:
            load_checkpoint(model, config.model.checkpoint)

            print('classic model:')
            show_params(model, verbose=config.debug.show_models)

        freeze_backbone(model)
        opt = base_optim(
            get_grad_params(model.backbone),
            lr=1e-6  # BB already optimized
        )
        scheduler = None
    elif config.pipeline.model == 'ours':
        # freeze_backbone(model)
        if config.ours.data.using_heatmaps:
            roto_net = None  # todo
        else:
            if config.ours.triangulate == 'master':
                roto_net = Cam2camNet
            elif config.ours.triangulate == 'world':
                roto_net = RotoTransNet

        model = roto_net(config).to(device)  # todo DistributedDataParallel
        if config.pipeline.model == 'ours':
            if config.ours.model.init_weights:
                load_checkpoint(model, config.ours.model.checkpoint)
            else:
                print('our model:')
                show_params(model, verbose=config.debug.show_models)

        params = [
            {
                'params': get_grad_params(model),
                'lr': config.ours.opt.lr  # try me: 1e-4 seems too much larger, NaN when triangulating
            }
        ]

        if not config.ours.data.using_gt:  # predicting KP and HM -> need to opt
            print('using predicted KPs => adding model.backbone to grad ...')
            params.append(
                {
                    'params': get_grad_params(model.backbone),
                    'lr': 1e-6  # BB already optimized
                }
            )

        opt = base_optim(params, weight_decay=config.ours.opt.weight_decay)
        opts = config.ours.opt.scheduler
        scheduler = ReduceLROnPlateau(
            opt,
            factor=opts.factor,  # new lr = x * lr
            patience=opts.patience,  # n max iterations since optimum
            # threshold=42,  # no matter what, do lr decay
            mode='min',
            cooldown=int(opts.patience * 0.05),  # 5%
            min_lr=opts.min_lr,
            verbose=True
        )  # https://www.mayoclinic.org/healthy-lifestyle/weight-loss/in-depth/weight-loss-plateau/art-20044615
    elif config.pipeline.model == 'canonpose':
        model = CanonPose(
            inner_size=config.canonpose.model.inner_size,
            n_joints=config.canonpose.data.n_joints
        ).to(device)
        if config.canonpose.model.init_weights:
            load_checkpoint(model, config.ours.model.checkpoint)
        else:
            print('canonpose model:')
            show_params(model, verbose=config.debug.show_models)

        params = [
            {
                'params': get_grad_params(model),
                'lr': config.canonpose.opt.lr
            }
        ]
        opt = optim.Adam(
            params,
            lr=config.canonpose.opt.lr,
            weight_decay=config.canonpose.opt.weight_decay
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 90], gamma=0.1
        )

    return model, opt, scheduler
