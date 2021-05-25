from torch.nn.parallel import DistributedDataParallel

from mvn.pipeline.core import one_epoch
from mvn.pipeline.setup import setup_dataloaders, setup_experiment, build_env
from mvn.utils.minimon import MiniMon


def do_eval(config_path, logdir, config, device, is_distributed, master):
    model, cam2cam_model, criterion, opt, scheduler = build_env(config, device)
    if is_distributed:  # multi-gpu
        model = DistributedDataParallel(model, device_ids=[device])

    _, val_dataloader, _ = setup_dataloaders(config, distributed_train=is_distributed)  # ~ 0 seconds

    if master:
        experiment_dir = setup_experiment(
            config_path, logdir, config, type(model).__name__
        )
    else:
        experiment_dir = None

    minimon = MiniMon()

    minimon.enter()
    one_epoch(
        model, criterion, opt, scheduler, config, val_dataloader, device, 0,
        minimon, is_train=False, master=master, experiment_dir=experiment_dir, cam2cam_model=cam2cam_model
    )
    minimon.leave('do eval')

    if master:
        minimon.print_stats(as_minutes=False)
