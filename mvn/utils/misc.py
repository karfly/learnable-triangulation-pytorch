import os
import gc
import yaml
import json
from sys import getsizeof
from types import ModuleType, FunctionType
from gc import get_referents
import numpy as np
import inspect, re

import torch


# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST_FROM_SIZE = type, ModuleType, FunctionType  # from SO


def config_to_str(config):
    return yaml.dump(yaml.safe_load(json.dumps(config)))  # fuck yeah


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_gradient_norm(named_parameters):
    total_norm = 0.0
    for name, p in named_parameters:
        # print(name)
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** (1. / 2)

    return total_norm


def normalize_transformation(feature_range):
    def _f(x):
        shape = x.shape

        m = x.min()
        M = x.max()

        x = (x - m) / (M - m) * (feature_range[1] - feature_range[0]) + feature_range[0]

        x = x.reshape(shape)  # original size
        return x

    return _f


def get_size(obj):
    """sum size of object & members."""

    if isinstance(obj, BLACKLIST_FROM_SIZE):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))

    seen_ids = set()
    size = 0
    objects = [obj]

    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST_FROM_SIZE) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)

    return size  # bytes


def flush_cache(in_torch=True, with_garbage_collector=True, verbose=False):
    if with_garbage_collector:
        gc.collect()

    if in_torch:
        torch.cuda.empty_cache()

    if verbose:
        print(torch.cuda.memory_summary(device=None, abbreviated=False))


def is_distributed():
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    return True


def is_master():
    if is_distributed():
        return int(os.environ["RANK"]) == 0  # notation: proc #0 is THE MASTER #fuckGithub

    return True  # just 1 process


def drop_na(x):
    return np.float32(list(filter(
        lambda x: np.isfinite(x), x
    )))


def find_min(lst):
    m = np.min(drop_na(lst))
    return m, list(lst).index(m)


# todo use pythonic logger
def live_debug_log(tag, message, master_only=True):
    can_print = (not master_only) or (master_only and is_master())
    if can_print:
        print('#[{}]: {}'.format(
            tag, message
        ))


def get_others():
    return [
        [ 1, 2, 3 ],
        [ 0, 2, 3 ],
        [ 0, 1, 3 ],
        [ 0, 1, 2 ],
    ]  # todo auto


def get_pairs():
    return [
        [ (0, 1), (0, 2), (0, 3) ],
        [ (1, 0), (1, 2), (1, 3) ],
        [ (2, 0), (2, 1), (2, 3) ],
        [ (3, 0), (3, 1), (3, 2) ],
    ]  # todo auto


def get_i_from_pair(i, j):
    return (
        i,
        j - 1 if j > i else j
    )


def get_inverse_i_from_pair(i, j):
    return get_i_from_pair(j, i)


def get_master_pairs():
    return [
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [2, 0, 1, 3],
        [3, 0, 1, 2]
    ]  # todo auto


# https://stackoverflow.com/a/592849
def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        print(m)
        # if m:
        #     return m.group(1)

    return None


def get_exception_trace(ex):
    trace = []
    tb = ex.__traceback__
    while tb is not None:
        trace.append({
            "filename": tb.tb_frame.f_code.co_filename,
            "name": tb.tb_frame.f_code.co_name,
            "lineno": tb.tb_lineno
        })

        tb = tb.tb_next

    return {
        'type': type(ex).__name__,
        'message': str(ex),
        'trace': trace
    }