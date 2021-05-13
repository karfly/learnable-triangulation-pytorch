import os
import argparse

import torch

from mvn.pipeline.training import do_train
from mvn.utils.cfg import load_config
from mvn.utils.misc import is_master, is_distributed


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


def init_distributed(args):
    if not is_distributed():
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True


def main(args):
    print('# available GPUs: {:d}'.format(torch.cuda.device_count()))

    is_distributed = init_distributed(args)
    master = is_master()
    device = torch.device(args.local_rank) if is_distributed else torch.device(0)
    print('using dev {}'.format(device))

    config = load_config(args.config)
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    do_train(args.config, args.logdir, config, device, is_distributed, master)
    # todo do eval based on args


if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))

    main(args)
