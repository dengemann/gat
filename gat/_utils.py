# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: Simplified BSD

# This module is a permanent WIP

import numpy as np


def drop_bad_epochs(epochs, reject=None, flat=None):
    # XXX to be removed once #1900 is merged
    import numpy as np
    epochs.reject = reject
    epochs.flat = flat
    epochs.reject_tmin = None
    epochs.reject_tmax = None
    epochs._reject_setup()
    drop_inds = list()
    if epochs.reject is not None or epochs.flat is not None:
        for i_epoch, epoch in enumerate(epochs):
            is_good, chan = epochs._is_good_epoch(epoch)
            if not is_good:
                drop_inds.append(i_epoch)
                epochs.drop_log[i_epoch].extend(chan)
    if drop_inds:
        select = np.ones(len(epochs.events), dtype=np.bool)
        select[drop_inds] = False
        epochs.events = epochs.events[select]
        epochs._data = epochs._data[select]
        epochs.selection[select]
    return epochs


def resample_epochs(epochs, sfreq):
    """Fast MNE epochs resampling"""
    # from librosa import resample
    # librosa.resample(channel, o_sfreq, sfreq, res_type=res_type)
    from scipy.signal import resample

    # resample
    epochs._data = resample(
        epochs._data, epochs._data.shape[2] / epochs.info['sfreq'] * sfreq,
        axis=2)
    # update metadata
    epochs.info['sfreq'] = sfreq
    epochs.times = (np.arange(epochs._data.shape[2],
                              dtype=np.float) / sfreq + epochs.times[0])
    return epochs


def decim(inst, decim):
    """Fast MNE object decimation"""
    from mne.io.base import _BaseRaw
    from mne.epochs import _BaseEpochs
    if isinstance(inst, _BaseRaw):
        inst._data = inst._data[:, ::decim]
        inst.info['sfreq'] /= decim
        inst._first_samps /= decim
        inst._last_samps /= decim
    elif isinstance(inst, _BaseEpochs):
        inst._data = inst._data[:, :, ::decim]
        inst.info['sfreq'] /= decim
        inst.times = inst.times[::decim]
    return inst


def Evokeds_to_Epochs(inst, info=None, events=None):
    """Convert list of evoked into single epochs

    Parameters
    ----------
    inst: list
        list of evoked objects.
    info : dict
        By default copy dict from inst[0]
    events : np.array (dims: n, 3)
    Returns
    -------
    epochs: epochs object"""
    from mne.epochs import EpochsArray
    from mne.evoked import Evoked

    if (
        not(isinstance(inst, list)) or
        not np.all([isinstance(x, Evoked) for x in inst])
    ):
        raise('inst mus be a list of evoked')

    # concatenate signals
    data = [x.data for x in inst]
    # extract meta data
    if info is None:
        info = inst[0].info
    if events is None:
        n = len(inst)
        events = np.c_[np.cumsum(np.ones(n)) * info['sfreq'],
                       np.zeros(n), np.ones(n)]

    return EpochsArray(data, info, events=events, tmin=inst[0].times.min())


def find_in_df(df, include, exclude=dict(), max_n=np.inf):
    """Find instance in pd.dataFrame that correspond to include and exlcuding
    criteria.

    Parameters
    ----------
    df : pd.dataFrame
    includes : list | dict()
    excludes : list | dict()
    Returns
    -------
    inds : np.array"""
    import random

    # Find included trials
    include_inds = _find_in_df(df, include)
    # Find excluded trials
    exclude_inds = _find_in_df(df, exclude)

    # Select condition
    inds = [i for i in include_inds if i not in exclude_inds]

    # reduce number or trials if too many
    if len(inds) > max_n:
        random.shuffle(inds)
        inds = inds[:max_n]

    return inds


def _find_in_df(df, le_dict):
    """Find all instances in pd dataframe that match one of the specified
    conditions"""
    inds = []
    for key in le_dict.keys():
        if type(le_dict[key]) not in (list, np.ndarray):
            le_dict[key] = [le_dict[key]]
        for value in le_dict[key]:
            for i in np.where(df[key] == value)[0]:
                inds.append(i)
    inds = np.unique(inds)
    return inds


class cluster_stat(dict):
    """ Cluster statistics """
    def __init__(self, insts, alpha=0.05, **kwargs):
        """
        Parameters
        ----------
        X : np.array (dims = n * space * time)
            data array
        alpha : float
            significance level

        Can take spatio_temporal_cluster_1samp_test() parameters.

        """
        from mne.stats import spatio_temporal_cluster_1samp_test

        # Convert lists of evoked in Epochs
        insts = [Evokeds_to_Epochs(i) if type(i) is list else i for i in insts]

        # Apply contrast: n * space * time
        X = np.array(insts[0]._data - insts[-1]._data).transpose([0, 2, 1])

        # Run stats
        self.T_obs_, clusters, p_values, _ = \
            spatio_temporal_cluster_1samp_test(X, out_type='mask', **kwargs)

        # Save sorted sig clusters
        inds = np.argsort(p_values)
        clusters = np.array(clusters)[inds, :, :]
        p_values = p_values[inds]
        inds = np.where(p_values < alpha)[0]
        self.sig_clusters_ = clusters[inds, :, :]
        self.p_values_ = p_values[inds]

        # By default, keep meta data from first epoch
        self.insts = insts
        self.times = self.insts[0].times
        self.info = self.insts[0].info
        self.ch_names = self.insts[0].ch_names

        return

    def _get_mask(self, i_clu):
        """
        Selects or combine clusters

        Parameters
        ----------
        i_clu : int | list | array
            cluster index. If list or array, returns average across multiple
            clusters.

        Returns
        -------
        mask : np.array
        space_inds : np.array
        times_inds : np.array
        """
        # Select or combine clusters
        if i_clu is None:
            i_clu = range(len(self.sig_clusters_))
        if isinstance(i_clu, int):
            mask = self.sig_clusters_[i_clu]
        else:
            mask = np.sum(self.sig_clusters_[i_clu], axis=0)

        # unpack cluster infomation, get unique indices
        space_inds = np.where(np.sum(mask, axis=0))[0]
        time_inds = np.where(np.sum(mask, axis=1))[0]

        return mask, space_inds, time_inds

    def plot_topo(self, i_clu=None, pos=None, **kwargs):
        """
        Plots fmap of one or several clusters.

        Parameters
        ----------
        i_clu : int
            cluster index

        Can take evoked.plot_topomap() parameters.

        Returns
        -------
        fig
        """
        from mne import find_layout
        from mne.viz import plot_topomap

        # Channel positions
        pos = find_layout(self.info).pos
        # create topomap mask from sig cluster
        mask, space_inds, time_inds = self._get_mask(i_clu)

        if pos is None:
            pos = find_layout(self.info).pos

        # plot average test statistic and mark significant sensors
        topo = self.T_obs_[time_inds, :].mean(axis=0)
        fig = plot_topomap(topo, pos, **kwargs)

        return fig

    def plot_topomap(self, i_clu=None, **kwargs):
        """
        Plots effect topography and highlights significant selected clusters.

        Parameters
        ----------
        i_clu : int
            cluster index

        Can take evoked.plot_topomap() parameters.

        Returns
        -------
        fig
        """
        # create topomap mask from sig cluster
        mask, space_inds, time_inds = self._get_mask(i_clu)

        # plot average test statistic and mark significant sensors
        evoked = self.insts[0].average()
        evoked.data = self.T_obs_.transpose()
        fig = evoked.plot_topomap(mask=np.transpose(mask), **kwargs)

        return fig

    def plot(self, plot_type='butterfly', i_clus=None, axes=None, show=True,
             **kwargs):
        """
        Plots effect time course and highlights significant selected clusters.

        Parameters
        ----------
        i_clus : None | list | int
            cluster indices
        plot_type : str
            'butterfly' to plot differential response across all channels
            'cluster' to plot cluster time course for each condition

        Can take evoked.plot() parameters.

        Returns
        -------
        fig
        """
        import matplotlib.pyplot as plt
        from mne.viz.utils import COLORS

        times = self.times * 1000

        # if axes is None:
        if True:
            fig = plt.figure()
            fig.add_subplot(111)
            axes = fig.axes[0]

        # By default, plot separate clusters
        if i_clus is None:
            if plot_type == 'butterfly':
                i_clus = [None]
            else:
                i_clus = range(len(self.sig_clusters_))
        elif isinstance(i_clus, int):
            i_clus = [i_clus]

        # Time course
        if plot_type == 'butterfly':
            # Plot butterfly of difference
            evoked = self.insts[0].average() - self.insts[1].average()
            fig = evoked.plot(axes=axes, show=False, **kwargs)
        elif plot_type == 'cluster':
            evokeds = [x.average() for x in self.insts]
            for i_clu in i_clus:
                _, space_inds, _ = self._get_mask(i_clu)
                for i_evo, evoked in enumerate(evokeds):
                    signal = np.mean(evoked.data[space_inds, :],
                                     axis=0)
                    _kwargs = kwargs.copy()
                    _kwargs['color'] = COLORS[i_evo % len(COLORS)]
                    axes.plot(times, signal, **_kwargs)

        # Significant times
        ymin, ymax = axes.get_ylim()
        for i_clu in i_clus:
            _, _, time_inds = self._get_mask(i_clu)
            sig_times = times[time_inds]

            fill_betweenx_discontinuous(axes, ymin, ymax, sig_times,
                                        freq=(self.info['sfreq'] / 1000),
                                        color='orange', alpha=0.3)

        axes.legend(loc='lower right')
        axes.set_ylim(ymin, ymax)

        # add information
        axes.axvline(0, color='k', linestyle=':', label='stimulus onset')
        axes.set_xlim([times[0], times[-1]])
        axes.set_xlabel('Time [s]')
        axes.set_ylabel('Evoked magnetic fields [fT]')

        if show:
            plt.show()

        return fig
