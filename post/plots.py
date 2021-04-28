import numpy as np
from matplotlib import pyplot as plt


def plot_metric(axis, metrics, label, xrange=None, ylim=[0, 200], color='black', legend_loc=None):
    if xrange is None:
        done = len(metrics)
        xrange = list(range(1, done + 1))

    axis.plot(
        xrange, metrics, label=label, color=color
    )

    if ylim:
        axis.set_ylim(ylim)

    if legend_loc:
        axis.legend(loc=legend_loc)


def plot_metrics(axis, train_metrics, eval_metrics, xrange=None, train_ylim=[0, 30], eval_ylim=[0, 100]):
    plot_metric(
        axis,
        train_metrics,
        'on TRAIN set (S1, S6, S7, S8)',
        xrange=xrange,
        ylim=train_ylim,
        color='red',
        legend_loc='upper left'
    )

    axis = axis.twinx()

    plot_metric(
        axis,
        eval_metrics,
        'on EVAL set (S9, S11)',
        xrange=xrange,
        ylim=eval_ylim,
        color='green',
        legend_loc='upper right'
    )


def plot_epochs(at_least_2_axis, epochs):
    axis = at_least_2_axis.ravel()[0]
    training_loss = [
        np.mean(epoch['training loss / batch'])
        for epoch in epochs
    ]

    plot_metric(
        axis,
        training_loss,
        'training loss',
        ylim=[0, 5],
        color='red',
        legend_loc='upper left'
    )
    axis.set_xlim([0, len(epochs) - 1])
    axis.set_ylabel('geodesic loss')
    axis.set_xlabel('# epoch')

    axis = axis.twinx()  # on the right

    plot_metric(
        axis,
        [epoch['training metrics'] for epoch in epochs],
        'training metrics',
        ylim=[0, 20],
        color='green',
        legend_loc='upper right'
    )
    axis.set_xlim([0, len(epochs) - 1])
    axis.set_ylabel('average MPJPE relative to pelvis (mm)')

    axis = at_least_2_axis.ravel()[1]
    axis.set_yticks([])
    axis.set_xlabel('# epoch')

    axis = axis.twinx()  # on the right

    plot_metric(
        axis,
        [epoch['eval metrics'] for epoch in epochs],
        'eval metrics',
        ylim=[1600, 5000],
        color='blue',
        legend_loc='upper right'
    )
    axis.set_xlim([0, len(epochs) - 1])
    axis.set_ylabel('average MPJPE relative to pelvis (mm)')


def make_axis_great_again(ax, title, xlim=None, ylim=None):
    if ylim:
        ax.set_ylim(ylim)
    
    if xlim:
        ax.set_xlim(xlim)

    ax.grid(True)
    ax.set_title(title)

    ax.set_ylabel('average MPJPE relative to pelvis (mm)')
    ax.set_xlabel('epoch')


def get_figsize(n_rows, n_cols, row_size=8, column_size=24):
    return (n_cols * column_size, n_rows * row_size)


def get_figa(n_rows, n_cols, heigth=8, width=24):
    fig_size = get_figsize(n_rows, n_cols, row_size=heigth, column_size=width)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=fig_size)
    return fig, ax
