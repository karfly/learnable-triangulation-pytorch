import os
import yaml
import json
import re
from sys import getsizeof

import torch


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


def get_size(x, mb=True):
    kb = getsizeof(x) / 1024.0
    
    if mb:
        return kb / 1024.0
    
    return kb
