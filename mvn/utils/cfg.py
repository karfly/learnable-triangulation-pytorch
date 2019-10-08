import yaml
from easydict import EasyDict as edict


def load_config(path):
    with open(path) as fin:
        config = edict(yaml.safe_load(fin))

    return config
