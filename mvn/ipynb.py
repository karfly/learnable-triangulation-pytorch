import torch
import numpy as np
from pathlib import Path

from mvn.utils import multiview, cfg
from mvn.utils.dicts import NestedNamespace
from mvn.models.utils import build_opt

from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss, element_weighted_loss


def get_args():
    args = NestedNamespace(
        dict(
            config='experiments/human36m/train/human36m_alg.yaml',
            eval=False,
            eval_dataset='val',
            local_rank=None, logdir='/home/stefano/_tmp/logs',
            seed=42
        )
    )

    print('# available GPUs: {:d}'.format(torch.cuda.device_count()))
    
    return args


def get_config(args, data_folder='/home/stefano/Scuola/tud/_classes/4/thesis/data/'):
    config = cfg.load_config(args.config)

    config.image_shape = [384, 384]

    config.debug.show_models = False
    config.debug.write_imgs = False
    config.debug.img_out = '/home/stefano/Scuola/tud/_classes/4/thesis/logs/imgs'
    config.debug.dump_checkpoints = False

    config.opt.n_epochs = 1
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    config.opt.batch_size = 8
    config.opt.val_batch_size = 16

    config.opt.loss_3d = False
    config.opt.loss_2d = not config.opt.loss_3d

    config.opt.torch_anomaly_detection = False

    config.model.init_weights = False  # there is no point in loading full module with a shitty GPU
    # config.model.checkpoint = data_folder + 'weights_alg.pth'  #  + 'weights_vol.pth'

    config.model.triangulate_in_world_space = False
    config.model.triangulate_in_cam_space = False
    config.model.cam2cam_estimation = True

    config.cam2cam.using_gt = True
    config.cam2cam.using_heatmaps = False  # KPs seem to work much better
    config.cam2cam.model.init_weights = False
    # config.cam2cam.model.checkpoint = '/home/stefano/Scuola/tud/_classes/thesis/milestones/06.05_12.05_solving_the_MLP_problem/weights_cam2cam_model.pth'

    config.cam2cam.model.name = 'kiss'
    config.cam2cam.model.n_inner_layers = 2
    config.cam2cam.vgg = 'smallest'
    config.cam2cam.batch_norm = False
    config.cam2cam.model.inner_size = 16
    config.cam2cam.loss.roto_weight = 0.0
    config.cam2cam.loss.trans_weight = 0.0
    config.cam2cam.loss.proj_weight = 1.0
    config.cam2cam.loss.loss_3d_weight = 0.0
    config.cam2cam.opt.lr = 2e-5

    config.model.backbone.init_weights = config.model.init_weights
    # config.model.backbone.checkpoint = data_folder + 'pose_resnet_4.5_pixels_human36m.pth'
    config.model.backbone.num_layers = 18  # very small BB
    config.model.backbone.num_deconv_filters = 32

    config.dataset.train.crop = not config.model.cam2cam_estimation  # doing resampling when estimating cam2cam => no crop
    config.dataset.train.h36m_root = data_folder + 'processed/'
    config.dataset.train.labels_path = data_folder + 'human36m-multiview-labels-GTbboxes.npy'
    config.dataset.train.num_workers = 0
    config.dataset.train.retain_every_n_frames_in_train = 10000  # 12 images when in full dataset

    config.dataset.val.crop = config.dataset.train.crop
    config.dataset.val.h36m_root = config.dataset.train.h36m_root
    config.dataset.val.labels_path = config.dataset.train.labels_path
    config.dataset.val.num_workers = 0
    config.dataset.val.retain_every_n_frames_in_test = 500  # 5 images when in full dataset
    
    return config


def build_labels(f_path, retain_every_n_frames, allowed_subjects=['S1', 'S6', 'S7', 'S8']):
    print('estimating dataset size ...')
    labels = np.load(f_path, allow_pickle=True).item()
    
    subjects = [
        labels['subject_names'].index(x)
        for x in allowed_subjects  # todo solve missing images in 'S5'
    ]
    
    mask = np.isin(labels['table']['subject_idx'], subjects, assume_unique=True)
    indices = np.nonzero(mask)[0][::retain_every_n_frames]
    
    print('  ... available subjects {} and subsampling 1/{:d} => {:d} available frames'.format(
        allowed_subjects,
        retain_every_n_frames,
        len(indices)
    ))
    
    return labels, mask, indices
