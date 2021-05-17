import torch

from mvn.utils.misc import flush_cache

from mvn.ipynb import get_args, get_config
from mvn.pipeline.training import do_train

flush_cache()

is_distributed = False
master = True
device = torch.device(0)
print('using dev {}'.format(device))

args = get_args()

# just to get a feeling of the dataset
# labels, mask, indices = build_labels(config.dataset.train.labels_path, 10000)
# labels, mask, indices = build_labels(config.dataset.train.labels_path, 10, allowed_subjects=['S9', 'S11'])

config = get_config(args)

try:
    do_train(None, None, config, device, is_distributed, master)
except ZeroDivisionError:
    print('did you set a breakpoint?')
