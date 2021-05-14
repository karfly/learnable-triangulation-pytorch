import numpy as np
from matplotlib import pyplot as plt

from mvn.utils.misc import find_min, drop_na, normalize_transformation

LOSS_C = ['violet', 'gold', 'lawngreen', 'red', 'brown', 'gray', 'black']


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


def plot_metric(axis, metrics, label, xrange=None, ylim=None, color='black', alpha=1.0, legend_loc=None, show_min=False, marker=',', verbose=True):
    if xrange is None:
        done = len(metrics)
        xrange = list(range(done))

    axis.plot(
        xrange, metrics, label=label, color=color, marker=marker, alpha=alpha
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


# todo split into `plot_losses / plot_metrics`
def plot_epochs(axis, epochs, xrange, train_metric_ylim=[0, 1], eval_metric_ylim=[0, 1], normalize_loss=None, title=None, metric_ylabel=None, xlabel='# epoch'):
    loss_keys = list(filter(
        lambda x: 'loss / batch' in x,  # and 'training' not in x,
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
        nan = np.mean(drop_na(loss_history))
        loss_history = np.nan_to_num(loss_history, nan=nan)

        if np.mean(loss_history) > 1e-2:
            _min, _max = np.min(drop_na(loss_history)), np.max(
                drop_na(loss_history))
            _last = loss_history[-1]
            label = '{} = {:.0f} ({:.0f} / {:.0f})'.format(
                key.replace('loss / batch', '').strip(), _last, _min, _max
            )
            loss_history = normalize_transformation(normalize_loss)(
                loss_history) if normalize_loss else loss_history

            plot_metric(
                axis,
                loss_history,
                label=label,
                xrange=xrange,
                color=color,
                alpha=0.8 if 'total' in label else 0.3,
                legend_loc='lower left',
                show_min=False,
                marker='+',
                verbose=False
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
        marker='o',
        verbose=False
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
        marker='o',
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
            ymin=0, ymax=1,
            label='new lr: {:.3E}'.format(new_lr),
            color='magenta',
            linestyle=':',
            alpha=0.2
        )

    # axis.legend(loc='upper center')


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
