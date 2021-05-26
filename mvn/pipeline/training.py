import os
import torch

from torch.nn.parallel import DistributedDataParallel

from mvn.utils.misc import live_debug_log
from mvn.pipeline.core import one_epoch
from mvn.pipeline.setup import setup_dataloaders, setup_experiment, build_env
from mvn.utils.minimon import MiniMon


def do_train(config_path, logdir, config, device, is_distributed, master):
    _iter_tag = 'do_train'
    model, cam2cam_model, criterion, opt, scheduler = build_env(config, device)
    if is_distributed:  # multi-gpu
        model = DistributedDataParallel(model, device_ids=[device])

    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)  # ~ 0 seconds

    if master:
        experiment_dir = setup_experiment(
            config_path, logdir, config, type(model).__name__
        )
    else:
        experiment_dir = None

    minimon = MiniMon()

    for epoch in range(config.opt.n_epochs):  # training
        live_debug_log(_iter_tag, 'epoch {:4d} has started!'.format(epoch))

        if train_sampler:  # None when NOT distributed
            train_sampler.set_epoch(epoch)

        minimon.enter()
        one_epoch(
            model, criterion, opt, scheduler, config, train_dataloader, device, epoch,
            minimon, is_train=True, master=master, experiment_dir=experiment_dir, cam2cam_model=cam2cam_model
        )
        minimon.leave('do train')

        minimon.enter()
        one_epoch(
            model, criterion, opt, scheduler, config, val_dataloader, device, epoch,
            minimon, is_train=False, master=master, experiment_dir=experiment_dir, cam2cam_model=cam2cam_model
        )
        minimon.leave('do eval')

        if master and experiment_dir and config.debug.dump_checkpoints:
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)

            if epoch % config.opt.save_every_n_epochs == 0:
                if config.model.cam2cam_estimation:
                    torch.save(cam2cam_model.state_dict(), os.path.join(checkpoint_dir, "cam2cam_model.pth"))

                    if not config.cam2cam.using_gt:  # model was actually trained
                        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights_model.pth"))
                else:  # usual algebraic / vol model
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights_model.pth"))

        train_time_avg = 'do train'
        train_time_avg = minimon.store[train_time_avg].get_avg()

        val_time_avg = 'do eval'
        val_time_avg = minimon.store[val_time_avg].get_avg()

        epoch_time_avg = train_time_avg + val_time_avg
        epochs_in_1_hour = 60 * 60 / epoch_time_avg
        epochs_in_1_day = 24 * 60 * 60 / epoch_time_avg
        message = 'epoch time ~ {:.1f}" => {:.0f} epochs / hour, {:.0f} epochs / day'.format(epoch_time_avg, epochs_in_1_hour, epochs_in_1_day)
        live_debug_log(_iter_tag, message)

        live_debug_log(_iter_tag, 'epoch {:4d} complete!'.format(epoch))

    if master:
        minimon.print_stats(as_minutes=False)
