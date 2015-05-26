# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: Simplified BSD
import numpy as np
import matplotlib.pyplot as plt


def plot_eb(x, y, yerr, ax=None, alpha=0.3, color=None, line_args=dict(),
            err_args=dict()):
    """
    Parameters
    ----------
    x : list | np.array()
    y : list | np.array()
    yerr : list | np.array() | float
    ax
    alpha
    color
    line_args
    err_args

    Returns
    -------
    ax

    Adapted from http://tonysyu.github.io/plotting-error-bars.html#.VRE9msvmvEU
    """
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y,  color=color, **line_args)
    ax.fill_between(x, ymax, ymin, alpha=alpha, color=color, **err_args)

    return ax


def fill_betweenx_discontinuous(ax, ymin, ymax, x, freq=1, **kwargs):
    """Fill betwwen x even if x is discontinuous clusters
    Parameters
    ----------
    ax : axis
    x : list

    Returns
    -------
    ax : axis
    """
    x = np.array(x)
    min_gap = (1.1 / freq)
    while np.any(x):
        # If with single time point
        if len(x) > 1:
            xmax = np.where((x[1:] - x[:-1]) > min_gap)[0]
        else:
            xmax = [0]

        # If continuous
        if not np.any(xmax):
            xmax = [len(x) - 1]

        ax.fill_betweenx((ymin, ymax), x[0], x[xmax[0]], **kwargs)

        # remove from list
        x = x[(xmax[0] + 1):]
    return ax


def plot_mean_pred_diagonal(gat_list, y=None, ax=None, colors=None,
                            chance=None, show=True):
    from gat.utils import mean_y_pred
    import matplotlib.colors as mcol
    from mne.decoding import GeneralizationAcrossTime
    if isinstance(gat_list, GeneralizationAcrossTime):
        gat_list = [gat_list]

    times = gat_list[0].train_times_['times']

    preds_list = list()
    for gat in gat_list:
        preds = np.squeeze(np.transpose(mean_y_pred(gat, y), [2, 0, 1, 3]))
        preds_list.append([np.diagonal(pred).T for pred in preds])
    preds_list = np.transpose(preds_list, [1, 0, 2])

    if ax is None:
        fig, ax = plt.subplots(1)

    if colors is None:
        cmap = mcol.LinearSegmentedColormap.from_list('RdPuBu', ['r', 'b'])
        colors = [cmap(i) for i in np.linspace(0, 1, len(preds))]

    for preds, color in zip(preds_list, colors):
        plot_eb(times, np.mean(preds, axis=0),
                np.std(preds, axis=0) / np.sqrt(preds.shape[1]),
                ax=ax, color=color)
    if chance is not None:
        ax.axhline(chance, color='k')

    if show:
        plt.show()

    return fig


def plot_mean_pred(gat_list, y=None, ax=None, colors=None, show=True,
                   levels=[.10, np.inf], alpha=1., **kwargs):
    """WIP: only works for chance at .5"""
    from gat.utils import mean_y_pred
    import matplotlib.colors as mcol
    from mne.decoding import GeneralizationAcrossTime
    if isinstance(gat_list, GeneralizationAcrossTime):
        gat_list = [gat_list]

    preds_list = list()
    for gat in gat_list:
        preds = mean_y_pred(gat, y)
        preds_list.append(np.squeeze(preds))
    preds_list = np.mean(preds_list, axis=0).transpose(2, 0, 1)

    if colors is None:
        cmap = mcol.LinearSegmentedColormap.from_list('RdPuBu', ['r', 'b'])
        colors = [cmap(i) for i in np.linspace(0, 1, len(preds))]

    xx, yy = np.meshgrid(gat.train_times_['times'],
                         gat.test_times_['times'][0],
                         copy=False, indexing='xy')

    if levels is not None:
        if ax is None:
            fig, ax = plt.subplots(1)
        for pred, color in zip(preds_list, colors):
            ax.contour(xx, yy, abs(pred - .5), levels=levels,
                       colors=[color])
            ax.contourf(xx, yy, abs(pred - .5), levels=levels, colors=[color],
                        alpha=.05)
        ax.axvline(0, color='k')
        ax.axhline(0, color='k')
    else:
        if ax is None:
            fig, axs = plt.subplots(1, len(preds_list))
            kwargs_ = kwargs.copy()
            if 'show' in kwargs_.keys():
                kwargs_.pop('show')
            for pred, color, ax in zip(preds_list, colors, axs):
                gat.scores_ = pred
                gat.plot(ax=ax, show=False, **kwargs_)
            ax = axs
    if show is True:
        plt.show()
    return fig


def plot_diagonal(gat_list, significance=None, ax=None, color='blue',
                  show=True, **kwargs):
    from mne.decoding import GeneralizationAcrossTime

    if isinstance(gat_list, GeneralizationAcrossTime):
        gat_list = [gat_list]
    scores = [gat.scores_ for gat in gat_list]

    gat = gat_list[0]
    gat.scores_ = np.mean(scores, axis=0)

    if ax is None:
        fig, ax = plt.subplots(1)
    fig = gat.plot_diagonal(show=False, ax=ax, **kwargs)
    ymin, ymax = ax.get_ylim()

    scores_diag = np.array([np.diag(s) for s in scores])
    times = gat.train_times_['times']

    plot_eb(times,
            np.mean(scores_diag, axis=0),
            np.std(scores_diag, axis=0) / np.sqrt(len(scores)),
            ax=ax, color=color)

    if significance is not None:
        ymin, ymax = ax.get_ylim()
        times = gat.train_times_['times']
        sig_times = times[np.where(np.diag(significance))[0]]
        sfreq = (times[1] - times[0]) / 1000
        fill_betweenx_discontinuous(ax, ymin, ymax, sig_times, freq=sfreq,
                                    color='gray')

    if show is True:
        plt.show()

    return fig
