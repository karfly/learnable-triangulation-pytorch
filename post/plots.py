import numpy as np
from matplotlib import pyplot as plt

from mvn.utils.misc import find_min, drop_na, normalize_transformation

LOSS_C = ['magenta', 'gold', 'lawngreen', 'red']


def plot_SOTA(axis, _xrange):
    # show original paper results
    axis.hlines(
        21.3, xmin=_xrange[0], xmax=_xrange[-1],
        color='blue', linestyle=':', label='algebraic SOTA = 21.3'
    )

    # I'm using algebraic so volumetric SOTA may be misleading
    # axis.hlines(
    #     13.7, xmin=_xrange[0], xmax=_xrange[-1],
    #     color='green', linestyle=':', label='volumetric (softmax) SOTA = 13.7'
    # )


def plot_metric(axis, metrics, label, xrange=None, ylim=None, color='black', legend_loc=None, show_min=False, marker=',', verbose=True):
    if xrange is None:
        done = len(metrics)
        xrange = list(range(done))

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
    legend_loc = 'upper left'
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

    legend_loc = 'upper right'
    _xrange = plot_metric(
        axis,
        eval_metrics,
        'on EVAL set (S9, S11)',
        xrange=xrange,
        ylim=eval_ylim,
        color='green',
        legend_loc=legend_loc
    )
    plot_SOTA(axis, _xrange)
    axis.legend(loc=legend_loc)


def plot_epochs(axis, epochs, xrange, train_metric_ylim=[0, 1], eval_metric_ylim=[0, 1], normalize_loss=None, title=None, metric_ylabel=None, xlabel='# epoch'):
    loss_keys = list(filter(
        lambda x: 'loss / batch' in x and 'training' not in x,
        epochs[0].keys()
    ))
    if len(loss_keys) == 0:  # at least just show the training loss
        loss_keys = ['training loss / batch']
        colors = ['red']
    else:
        colors = LOSS_C

    for key, color in zip(loss_keys, colors):
        loss_history = np.float32([
            np.mean(epoch[key])
            for epoch in epochs
        ])

        if np.mean(loss_history) > 0.0:
            _min, _max = np.min(drop_na(loss_history)), np.max(
                drop_na(loss_history))
            _last = loss_history[-1]
            label = '{} (min = {:.1f}, max = {:.1f}, last = {:.1f})'.format(
                key.replace('loss / batch', '').strip(), _min, _max, _last
            )
            loss_history = normalize_transformation(normalize_loss)(
                loss_history) if normalize_loss else loss_history

            plot_metric(
                axis,
                loss_history,
                label=label,
                xrange=xrange,
                color=color,
                legend_loc='lower left',
                show_min=False,
                marker='+'
            )

    axis.set_xlim([0, len(epochs) - 1])
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.set_ylabel('loss')

    axis = axis.twinx()  # on the right

    legend_loc = 'upper right'
    plot_metric(
        axis,
        [epoch['training metrics'] for epoch in epochs],
        'training metrics',
        xrange=xrange,
        ylim=train_metric_ylim,
        color='aquamarine',
        legend_loc=legend_loc,
        show_min=True,
        marker='o'
    )

    plot_metric(
        axis,
        [epoch['eval metrics'] for epoch in epochs],
        'eval metrics',
        xrange=xrange,
        ylim=eval_metric_ylim,
        color='blue',
        legend_loc=legend_loc,
        show_min=True,
        marker='o'
    )

    # plot_SOTA(axis, [0, len(epochs) - 1])

    axis.legend(loc=legend_loc)
    axis.set_xlim([0, len(epochs) - 1])
    axis.set_ylabel(metric_ylabel)


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
