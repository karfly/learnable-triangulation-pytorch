import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

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


def plot_stuff(axis, stuff, label, xrange=None, ylim=None, color='black', alpha=1.0, legend_loc=None, show_min=False, marker=',', verbose=True):
    if xrange is None:
        done = len(stuff)
        xrange = list(range(done))

    if len(xrange) > len(stuff):
        xrange = xrange[:len(stuff)]

    axis.plot(
        xrange, stuff, label=label, color=color,
        marker=marker, markersize=5,
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
        alpha=0.9 if 'total' in label else 0.3,
        legend_loc=legend_loc,
        show_min=False,
        marker='o',
        verbose=False
    )


def plot_losses(axis, epochs, xrange, normalize_loss=None, title=None, xlabel='# epoch'):
    loss_keys = list(filter(
        lambda x: 'loss / batch' in x,  # and 'training' not in x,
        epochs[0].keys()
    ))
    if len(loss_keys) == 0:  # at least just show the training loss
        loss_keys = ['training loss / batch']
        colors = ['red']
    else:
        colors = [
            'tomato',
            'forestgreen',
            'lime',
            'maroon',
            'royalblue',
            'darkviolet',
            'fuchsia',
            'gray'
        ]

    loss_keys = [
        'geodesic',
        'L2 on T',
        'L2 proj',
        'L2 on 3D',
        # 'total',
        'self-consistency ext',
        'self-consistency P',
    ]  # forced
    multipliers = [
        3e1,
        3e-1,
        1e-2,
        1.5e-1,
        # 1e-2,
        5e-1,
        1e-2,
    ]

    for key, color, multip in zip(loss_keys, colors, multipliers):
        key += ' loss / batch'
        if key in epochs[0]:  # be sure to plot something that exists, we are not in QM
            loss_history = np.float32([
                np.mean(epoch[key])
                for epoch in epochs
            ])
            nan = np.mean(drop_na(loss_history))
            loss_history = np.nan_to_num(loss_history, nan=nan)

            if np.mean(loss_history) > 1e-2:
                _min, _max = np.min(drop_na(loss_history)), np.max(drop_na(loss_history))
                _last = loss_history[-1]
                label = '{} = {:.1f} [{:.1f}, {:.1f}]'.format(
                    key.replace('loss / batch', '').strip(),
                    _last, _min, _max
                )

                plot_loss(
                    axis,
                    loss_history * multip,
                    label,
                    xrange,
                    color
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
    marker = '+'

    metrics = drop_na(map(lambda x: x['training metrics'], epochs))
    plot_stuff(
        axis,
        metrics,
        'training metrics = {:.1f}'.format(metrics[-1]),
        xrange=xrange,
        ylim=train_metric_ylim,
        color='aquamarine',
        alpha=1.0,
        legend_loc=legend_loc,
        show_min=True,
        marker=marker,
        verbose=False
    )

    metrics = drop_na(map(lambda x: x['eval metrics'], epochs))
    plot_stuff(
        axis,
        metrics,
        'eval metrics = {:.1f}'.format(metrics[-1]),
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
            alpha=0.3
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
