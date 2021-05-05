import numpy as np
from matplotlib import pyplot as plt

from mvn.utils.misc import find_min, drop_na, normalize_transformation

LOSS_C = ['magenta', 'gold', 'lawngreen', 'red']


def plot_metric(axis, metrics, label, xrange=None, ylim=[0, 200], color='black', legend_loc=None, show_min=False, marker=',', verbose=True):
    if xrange is None:
        done = len(metrics)
        xrange = list(range(done))
    else:
        xrange = list(range(xrange[0], xrange[-1] + 1))

    axis.plot(
        xrange, metrics, label=label, color=color, marker=marker
    )

    if show_min:
        m, m_i = find_min(metrics)
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
            np.min(drop_na(metrics)), np.max(drop_na(metrics)),
            xrange[0], xrange[-1]
        ))

    return xrange


def plot_metrics(axis, train_metrics, eval_metrics, xrange=None, train_ylim=[0, 30], eval_ylim=[0, 100]):
    legend_loc='upper left'
    _xrange = plot_metric(
        axis,
        train_metrics,
        'on TRAIN set (S1, S6, S7, S8)',
        xrange=xrange,
        ylim=train_ylim,
        color='red',
        legend_loc=legend_loc
    )

    axis = axis.twinx()

    legend_loc='upper right'
    _xrange = plot_metric(
        axis,
        eval_metrics,
        'on EVAL set (S9, S11)',
        xrange=xrange,
        ylim=eval_ylim,
        color='green',
        legend_loc=legend_loc
    )

    # show original paper results
    axis.hlines(
        21.3, xmin=_xrange[0], xmax=_xrange[-1],
        color='green', linestyle=':', label='algebraic SOTA = 21.3'
    )

    # I'm using algebraic so volumetric SOTA may be misleading
    # axis.hlines(
    #     13.7, xmin=_xrange[0], xmax=_xrange[-1],
    #     color='green', linestyle=':', label='volumetric (softmax) SOTA = 13.7'
    # )

    axis.legend(loc=legend_loc)


def plot_epochs(axis, epochs, train_metric_ylim=[0, 1], eval_metric_ylim=[0, 1], loss_ylim=[0, 1], title=None, metric_ylabel=None):
    loss_keys = filter(
        lambda x: 'loss / batch' in x and 'training' not in x,
        epochs[0].keys()
    )

    for key, color in zip(loss_keys, LOSS_C):
        loss_history = np.float32([
            np.mean(epoch[key])
            for epoch in epochs
        ])

        _min, _last = np.min(drop_na(loss_history)), loss_history[-1]
        label = '{} (min = {:.1f}, last = {:.1f})'.format(
            key.replace('loss / batch', '').strip(), _min, _last
        )
        plot_metric(
            axis,
            normalize_transformation((0, 1))(loss_history),
            label=label,
            ylim=[0, 1],  # since it's normalized
            color=color,
            legend_loc='lower left',
            show_min=False,
            marker='+'
        )

    axis.set_xlim([0, len(epochs) - 1])
    axis.set_xlabel('# epoch')
    axis.set_title(title)
    axis.yaxis.set_ticks([])
    axis.set_ylabel('training losses')

    axis = axis.twinx()  # on the right

    plot_metric(
        axis,
        [epoch['training metrics'] for epoch in epochs],
        'training metrics',
        ylim=train_metric_ylim,
        color='aquamarine',
        legend_loc='upper right',
        show_min=True,
        marker='o'
    )

    plot_metric(
        axis,
        [epoch['eval metrics'] for epoch in epochs],
        'eval metrics',
        ylim=eval_metric_ylim,
        color='blue',
        legend_loc='upper right',
        show_min=True,
        marker='o'
    )
    axis.set_xlim([0, len(epochs) - 1])
    axis.set_ylabel(metric_ylabel)


def make_axis_great_again(ax, title=None, left_title=None, right_title=None, xlim=None, ylim=None):
    if ylim:
        ax.set_ylim(ylim)
    
    if xlim:
        ax.set_xlim(xlim)

    ax.grid(True)
    ax.set_xlabel('epoch')

    if title:
        ax.set_title(title)

    if left_title:
        ax.set_ylabel(left_title)

    if right_title:
        ax.twinx().set_ylabel(right_title)


def get_figsize(n_rows, n_cols, row_size=8, column_size=24):
    return (n_cols * column_size, n_rows * row_size)


def get_figa(n_rows, n_cols, heigth=8, width=24):
    fig_size = get_figsize(n_rows, n_cols, row_size=heigth, column_size=width)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=fig_size)
    return fig, ax
