from pathlib import Path
import torch

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # https://stackoverflow.com/a/56222305
from mvn.mini import get_config
from mvn.pipeline.setup import setup_dataloaders


def plot_kps(axis, kps, marker):
    axis.scatter(
        kps[:, 0],
        kps[:, 1],
        kps[:, 2],
        marker=marker
    )


def load_data(config, dumps_folder=Path('~/_tmp/').expanduser()):
    def _load(file_name):
        f_path = dumps_folder / file_name
        return torch.load(f_path).cpu().numpy()

    keypoints_3d_gt = _load('keypoints_3d_gt.trc')  # see `cam2cam:_save_stuff`
    keypoints_3d_pred = _load('keypoints_3d_pred.trc')
    indices = _load('batch_indexes')
    _, val_dataloader, _ = setup_dataloaders(config, distributed_train=False)  # ~ 0 seconds

    return keypoints_3d_gt, keypoints_3d_pred, indices, val_dataloader


def main(config):
    fig = plt.figure()
    axis = fig.gca(projection='3d')

    gts, pred, indices, dataloader = load_data(config)

    scalar_metric, full_metric = dataloader.dataset.evaluate(
        pred,
        indices_predicted=indices,  # [0, 3, 2, 4, 1]
        split_by_subject=True
    )  # (average 3D MPJPE (relative to pelvis), all MPJPEs)

    print(scalar_metric)  # full_metric

    plot_kps(axis, gts[2], 'o')  # todo also others
    plot_kps(axis, pred[2], '^')

    plt.show()


if __name__ == '__main__':
    config = get_config('experiments/human36m/train/human36m_alg.yaml')

    try:
        main(config)
    except ZeroDivisionError:
        print('Have you forgotten a breakpoint?')
