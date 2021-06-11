import numpy as np
from matplotlib import pyplot as plt

from mvn.utils.misc import find_min, drop_na


def plot_SOTA(axis, _xrange):
    # show original paper results
    diff = _xrange[-1] - _xrange[0]

    axis.hlines(
        21.3, xmin=_xrange[0] + diff / 2, xmax=_xrange[-1],
        color='blue', linestyle=':', label='algebraic SOTA = 21.3'
    )

    # I'm using algebraic so volumetric SOTA may be misleading
    # axis.hlines(
    #     13.7, xmin=_xrange[0], xmax=_xrange[-1],
    #     color='green', linestyle=':', label='volumetric (softmax) SOTA = 13.7'
    # )


def plot_stuff(axis, stuff, label, xrange=None, ylim=None, color='black', alpha=1.0, legend_loc=None, show_min=False, marker=',', linestyle='solid', verbose=True):
    if xrange is None:
        done = len(stuff)
        xrange = list(range(done))

    if len(xrange) > len(stuff):
        xrange = xrange[:len(stuff)]

    axis.plot(
        xrange, stuff, label=label, color=color,
        marker=marker, markersize=5, linestyle=linestyle,
        alpha=alpha
    )

    if show_min:
        m, m_i = find_min(stuff)
        axis.plot(
            xrange[m_i],
            m,
            marker='x', color='r', markersize=20,
            label='min = {:.1f}'.format(m)
        )

    if ylim:
        axis.set_ylim(ylim)

    if legend_loc:
        axis.legend(loc=legend_loc)

    if verbose:
        print('- plotted "{}" metrics [{:.1f}, {:.1f}] in epochs [{:.0f}, {:.0f}]'.format(
            label,
            np.min(drop_na(stuff)), np.max(drop_na(stuff)),
            xrange[0], xrange[-1]
        ))

    return xrange


def plot_loss(axis, loss_history, label, xrange, color):
    legend_loc = 'lower left'
    plot_stuff(
        axis,
        loss_history,
        label=label,
        xrange=xrange,
        color=color,
        alpha=0.9 if 'total' in label else 0.4,
        legend_loc=legend_loc,
        show_min=False,
        marker='o',
        linestyle='dashed',
        verbose=False
    )


def plot_losses(axis, epochs, xrange, normalize_loss=None, title=None, xlabel='# epoch'):
    loss_keys = list(filter(
        lambda x: 'loss / batch' in x,
        epochs[0].keys()
    ))

    n_max_losses = 10
    colors = plt.get_cmap('jet')(np.linspace(0, 1, n_max_losses))
    loss_plotters = {  # todo config file
        'total loss / batch': {
            'color': colors[0],
            'scaler': 1e-2,
            'show': False,
        },
        'R loss / batch': {
            'color': colors[1],
            'scaler': 3e-1,
            'show': True,
        },
        't loss / batch': {
            'color': colors[2],
            'scaler': 1e-3,
            'show': True,
        },
        'proj loss / batch': {
            'color': colors[3],
            'scaler': 5e-2,
            'show': False,
        },
        'world loss / batch': {
            'color': colors[4],
            'scaler': 1e-2,
            'show': True,
        },
        'self cam loss / batch': {
            'color': colors[5],
            'scaler': 5e-1,
            'show': False,
        },
        'self proj loss / batch': {
            'color': colors[7],  # colors[6] is yellow ...
            'scaler': 1e0,
            'show': True,
        },
        'self world loss / batch': {
            'color': colors[8],
            'scaler': 1e-3,
            'show': True,
        },
        'world struct loss / batch': {
            'color': colors[9],
            'scaler': 1e-3,
            'show': True,
        }
    }

    for key in loss_keys:
        if key in epochs[0]:  # be sure to plot something that exists, we are not in QM
            if key in loss_plotters and loss_plotters[key]['show']:
                loss_history = np.float32([
                    np.mean(epoch[key])
                    for epoch in epochs
                ])
                nan = np.mean(drop_na(loss_history))
                loss_history = np.nan_to_num(loss_history, nan=nan)

                if np.mean(loss_history) > 1e-2:  # non-trivial losses
                    _min, _max = np.min(drop_na(loss_history)), np.max(drop_na(loss_history))
                    # threshold = 1e5
                    # if _min < -threshold:
                    #     _min = float('-inf')
                    # if _max > threshold:
                    #     _max = float('+inf')

                    _last = loss_history[-1]
                    label = '{} = {:.1f} [{:.1f}, {:.1f}]'.format(
                        key.replace('loss / batch', '').strip(),
                        _last, _min, _max
                    )

                    scaler = loss_plotters[key]['scaler']
                    plot_loss(
                        axis,
                        loss_history * scaler,
                        label,
                        xrange,
                        loss_plotters[key]['color'],
                    )

    axis.set_xlim([xrange[0], xrange[-1]])
    axis.set_xlabel(xlabel)

    if title:
        axis.set_title(title)

    label = '{}loss'.format(
        '[{:.1f}, {:.1f}]-normalized '.format(
            normalize_loss[0], normalize_loss[1]
        ) if normalize_loss else ''
    )
    axis.set_ylabel(label)


def plot_metrics(axis, epochs, xrange, train_metric_ylim=[0, 1], eval_metric_ylim=[0, 1], metric_ylabel=None, with_SOTA=False):
    legend_loc = 'upper right'
    marker = ','

    metrics = np.float32(list(map(lambda x: x['training metrics (rel)'], epochs)))
    label = 'train rel MPJPE = {:.0f}'.format(metrics[-1])

    if 'training metrics (abs)' in epochs[-1]:
        last_abs_metrics = np.float32(epochs[-1]['training metrics (abs)'])
    else:
        last_abs_metrics = None
    
    # maybe pelvis is in origin ...
    if last_abs_metrics and abs(last_abs_metrics - metrics[-1]) > 1:
        label += ', abs MPJPE = {:.0f}'.format(
            last_abs_metrics
        )

    plot_stuff(
        axis,
        metrics,
        label,
        xrange=xrange,
        ylim=train_metric_ylim,
        color='aquamarine',
        alpha=1.0,
        legend_loc=legend_loc,
        show_min=True,
        marker=marker,
        verbose=False
    )

    metrics = np.float32(list(map(lambda x: x['eval metrics (rel)'], epochs)))
    label = 'eval rel MPJPE = {:.0f}'.format(metrics[-1])
    if 'eval metrics (abs)' in epochs[-1]:
        last_abs_metrics = np.float32(epochs[-1]['eval metrics (abs)'])
    else:
        last_abs_metrics = None

    # mabe pelvis is in origin ...
    if last_abs_metrics and abs(last_abs_metrics - metrics[-1]) > 1:
        label += ', abs MPJPE = {:.0f}'.format(
            last_abs_metrics
        )

    plot_stuff(
        axis,
        metrics,
        label,
        xrange=xrange,
        ylim=eval_metric_ylim,
        color='blue',
        alpha=1.0,
        legend_loc=legend_loc,
        show_min=True,
        marker=marker,
        verbose=False
    )

    plot_SOTA(axis, [xrange[0], xrange[-1]])

    axis.legend(loc=legend_loc)
    axis.set_xlim([xrange[0], xrange[-1]])
    axis.set_ylabel(metric_ylabel)


def plot_lr(axis, lr_reductions, batch_amount_per_epoch=8):
    for lr_reduction in lr_reductions:
        epoch = lr_reduction['epoch']
        batch_its = epoch * batch_amount_per_epoch

        new_lr = lr_reduction['lr']

        axis.vlines(
            x=batch_its,
            ymin=0, ymax=1e2,
            label='new lr: {:.3E}'.format(new_lr),
            color='magenta',
            linestyle=':',
            alpha=0.5
        )


def make_axis_great_again(ax, xlim=None, ylim=None, hide_y=False):
    if ylim:
        ax.set_ylim(ylim)

    if not (xlim is None):
        xlim = [xlim[0], xlim[-1]]
        ax.set_xlim(xlim)

    if hide_y:
        ax.yaxis.set_ticks([])


def get_figsize(n_rows, n_cols, row_size=8, column_size=24):
    return (n_cols * column_size, n_rows * row_size)


def get_figa(n_rows, n_cols, heigth=8, width=24):
    fig_size = get_figsize(n_rows, n_cols, row_size=heigth, column_size=width)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=fig_size)
    return fig, ax
