import argparse
import sys
import os

from mvn.utils.misc import flush_cache, get_exception_trace
from mvn.mini import get_config, build_labels

from main import main

flush_cache()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', type=str, required=False, help='Path, where config file is stored', default='experiments/human36m/train/human36m_alg.yaml'
    )
    parser.add_argument(
        '--eval', action='store_true', help='If set, then only evaluation will be done'
    )
    parser.add_argument(
        '--eval_dataset', type=str, help='Dataset split on which evaluate. Can be \'train\' and \'val\'', default='val'
    )
    parser.add_argument(
        '--local_rank', type=int, help='Local rank of the process on the node'
    )
    parser.add_argument(
        '--seed', type=int, required=False, help="Random seed for reproducibility", default=42
    )
    parser.add_argument(
        '--logdir', type=str, required=False, help='Path, where logs will be stored', default='/home/stefano/_tmp/logs'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print('args: {}'.format(args))

    config = get_config(args.config)

    try:
        main(args, config)
    except ZeroDivisionError as ex:
        print('Have you forgotten a breakpoint? ...')

        trace = get_exception_trace(ex)
        last = trace['trace'][-1]
        print('... maybe in {}:{} ?'.format(
            last['filename'], last['lineno']
        ))



    # just to get a feeling of the dataset
    # labels, mask, indices = build_labels(config.dataset.train.labels_path, 10000)
    # labels, mask, indices = build_labels(config.dataset.train.labels_path, 500, allowed_subjects=['S9', 'S11'])
