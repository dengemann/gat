# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: Simplified BSD
import numpy as np
import warnings
import matplotlib.pyplot as plt


def share_clim(axes, clim=None):
    """Share clim across multiple axes
    Parameters
    ----------
    axes : plt.axes
    clim : np.array | list, shape(2,), optional
        Defaults is min and max across axes.clim.
    """
    # Find min max of clims
    if clim is None:
        clim = list()
        for ax in axes:
            for im in ax.get_images():
                clim += np.array(im.get_clim()).flatten().tolist()
        clim = [np.min(clim), np.max(clim)]
    # apply common clim
    for ax in axes:
        for im in ax.get_images():
            im.set_clim(clim)
    plt.draw()


def plot_widths(xs, ys, widths, ax=None, color='b', xlim=None, ylim=None,
                **kwargs):
    if not (len(xs) == len(ys) == len(widths)):
        raise ValueError('xs, ys, and widths must have identical lengths')
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1)

    segmentx, segmenty = [xs[0]], [ys[0]]
    current_width = widths[0]
    for ii, (x, y, width) in enumerate(zip(xs, ys, widths)):
        segmentx.append(x)
        segmenty.append(y)
        if (width != current_width) or (ii == (len(xs) - 1)):
            ax.plot(segmentx, segmenty, linewidth=current_width, color=color,
                    **kwargs)
            segmentx, segmenty = [x], [y]
            current_width = width
    if xlim is None:
        xlim = [min(xs), max(xs)]
    if ylim is None:
        ylim = [min(ys), max(ys)]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ax if fig is None else fig


def plot_sem(x, y, **kwargs):
    """
    Parameters
    ----------
    x : list | np.array()
    y : list | np.array()
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
    m = np.nanmean(y, axis=0)
    std = np.nanstd(y, axis=0)
    n = y.shape[0] - np.sum(np.isnan(y), axis=0)

    return plot_eb(x, m, std / np.sqrt(n), **kwargs)


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


def plot_gat_times(gat_list, time, data='scores', chance=True,
                   color=None, title=None,
                   xmin=None, xmax=None, ymin=None, ymax=None,
                   ax=None, show=True,
                   xlabel='Time (s)', ylabel=None, legend=True, label=None):
    import matplotlib.colors as mcol
    if not isinstance(gat_list, list):
        gat_list = [gat_list]
    time_line_list = list()
    for gat in gat_list:
        # select data type
        if data == 'scores':
            if not hasattr(gat, 'scores_'):
                raise RuntimeError('Please score your data before trying to '
                                   'plot scores.')
            values = [gat.scores_]
            if label is None:
                label = ['Score']
            if ylabel is None:
                ylabel = 'Score'
        else:
            if not hasattr(gat, 'y_pred_'):
                raise RuntimeError('Please predict your data before trying to '
                                   'plot predictions.')
            values = gat.mean_ypred()
            if label is None:
                label = np.unique(gat.y_train_)
            if ylabel is None:
                ylabel = 'Prediction'

            # loop around pred_type
            values = np.transpose(values, [2, 0, 1, 3])

        # select time
        time_line = list()
        for value in values:
            time_line.append(_select_time_line(value, time, gat.train_times_,
                                               gat.test_times_))
        time_line_list.append(time_line)
    time_line_list = np.transpose(time_line_list, [1, 0, 2])
    times = [gat_list[0].test_times_['times']]
    times = np.linspace(np.min([np.min(ttimes) for ttimes in times]),
                        np.max([np.max(ttimes) for ttimes in times]),
                        np.shape(time_line_list)[2])

    # Plot
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if color is None:
        cmap = mcol.LinearSegmentedColormap.from_list('RdPuBu', ['r', 'b'])
        color = [cmap(i) for i in np.linspace(0, 1, len(time_line_list))]
    if isinstance(color, str):
        color = [color]
    if len(color) != len(gat_list):
        color = [color[idx % len(color)] for idx in range(len(time_line_list))]

    for time_line, col, lab in zip(time_line_list, color, label):
        plot_sem(times, time_line, ax=ax, color=col,
                 line_args=dict(label=label))

    if chance is True:
        chance = _get_chance_level(gat.scorer_, gat.y_train_)

    if chance is not False:
        ax.axhline(float(chance), color='k', linestyle='--',
                   label="Chance level")

    ax.axvline(0, color='k', label='')

    if title is not None:
        ax.set_title(title)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    if xmin is not None and xmax is not None:
        xmin = np.min(times)
        xmax = np.max(times)
    ax.set_xlim(xmin, xmax)
    if xlabel is not False:
        ax.set_xlabel(xlabel)
    if ylabel is not False:
        ax.set_ylabel(ylabel)
    if legend is True:
        ax.legend(loc='best')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(colors='dimgray')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.label.set_color('dimgray')
    ax.yaxis.label.set_color('dimgray')
    ax.spines['left'].set_color('dimgray')
    ax.spines['bottom'].set_color('dimgray')
    if show is True:
        plt.show()

    return fig if ax is None else ax.get_figure()


def _get_chance_level(scorer, y_train):
    # XXX JRK This should probably be solved within sklearn?
    if scorer.__name__ == 'accuracy_score':
        chance = np.max([np.mean(y_train == c) for c in np.unique(y_train)])
    elif scorer.__name__ == 'roc_auc_score':
        chance = 0.5
    else:
        chance = np.nan
        warnings.warn('Cannot find chance level from %s, specify chance'
                      ' level' % scorer.func_name)
    return chance


def _select_time_line(values, sel_time, train_times_, test_times_):
    # Detect whether gat is a full matrix or just its diagonal
    if np.all(np.unique([len(t) for t in test_times_['times']]) == 1):
        values_ = np.squeeze(values)
    elif sel_time == 'diagonal':
        # Get values from identical training and testing times even if GAT
        # is not square.
        values_ = np.zeros(len(values))
        for train_idx, train_time in enumerate(train_times_['times']):
            for test_times in test_times_['times']:
                # find closest testing time from train_time
                lag = test_times - train_time
                test_idx = np.abs(lag).argmin()
                # check that not more than 1 classifier away
                if np.abs(lag[test_idx]) > train_times_['step']:
                    value = np.nan
                else:
                    value = values[train_idx][test_idx]
                values_[train_idx] = value
    elif isinstance(sel_time, float):
        train_times = train_times_['times']
        idx = np.abs(train_times - sel_time).argmin()
        if train_times[idx] - sel_time > train_times_['step']:
            raise ValueError("No classifier trained at %s " % sel_time)
        values_ = values[idx]
    else:
        raise ValueError("train_time must be 'diagonal' or a float.")

    return values_


def plot_mean_pred(gat_list, y=None, ax=None, colors=None, show=True,
                   zscore=True, levels=[.10, np.inf], alpha=1.,  diagonal=True,
                   **kwargs):
    """WIP: only works for chance at .5"""
    import matplotlib.colors as mcol
    from mne.decoding import GeneralizationAcrossTime
    from gat.utils import GAT
    if isinstance(gat_list, GeneralizationAcrossTime):
        gat_list = [gat_list]

    gat_list = [GAT(gat) for gat in gat_list]

    if y is None or (isinstance(y, np.ndarray) and y.ndim == 1):
        y = y = [y for idx in gat_list]

    preds_list = list()
    for gat, y in zip(gat_list, y):
        if zscore:
            gat.y_pred_ = gat.zscore_ypred()
        preds = gat.mean_ypred(y=y)
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
        else:
            fig = ax.get_figure()
        for pred, color in zip(preds_list, colors):
            ax.contour(xx, yy, abs(pred - .5), levels=levels,
                       colors=[color])
            ax.contourf(xx, yy, abs(pred - .5), levels=levels, colors=[color],
                        alpha=.05)
        ax.axvline(0, color='k')
        ax.axhline(0, color='k')
        if diagonal:
            ax.plot(ax.get_xlim(), ax.get_ylim(), color='k')
    else:
        if ax is None:
            fig, ax = plt.subplots(1, len(preds_list))
        else:
            fig = ax[0].get_figure()
        kwargs_ = kwargs.copy()
        if 'show' in kwargs_.keys():
            kwargs_.pop('show')
        for pred, color, ax_ in zip(preds_list, colors, ax):
            gat.scores_ = pred
            gat.plot(ax=ax, show=False, **kwargs_)
            if diagonal:
                ax.plot(ax_.get_xlim(), ax_.get_ylim(), color='k')
    if show is True:
        plt.show()
    return fig
