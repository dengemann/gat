import copy
import numpy as np
from mne.decoding import GeneralizationAcrossTime


def subscore(gat, sel, y=None, scorer=None):
    """Subscores a GAT.

    Parameters
    ----------
        gat : GeneralizationAcrossTime
        sel : list or array, shape (n_predictions)
        y : None | list or array, shape (n_selected_predictions,)
            If None, y set to gat.y_true_. Defaults to None.

    Returns
    -------
    scores
    """
    gat_ = subselect_ypred(gat, sel)
    if scorer is not None:
        gat_.scorer = scorer
    return gat_.score(y=y)



def subselect_ypred(gat, sel):
    """Select subselection of y_pred_ of GAT.

    Parameters
    ----------
        gat : GeneralizationAcrossTime
        sel : list or array, shape (n_predictions)

    Returns
    -------
        new gat
    """
    import copy
    gat_ = copy.deepcopy(gat)
    # Subselection of trials
    for train in range(len(gat_.y_pred_)):
        for test in range(len(gat_.y_pred_[train])):
            gat_.y_pred_[train][test] = gat_.y_pred_[train][test][sel, :]
    gat_.y_train_ = gat_.y_train_[sel]
    return gat_


def mean_ypred(gat, y=None):
    """Provides mean prediction for each category.

    Parameters
    ----------
        gat : GeneralizationAcrossTime
        y : None | list or array, shape (n_predictions,)
            If None, y set to gat.y_train_. Defaults to None.

    Returns
    -------
    mean_y_pred : list of list of (float | array),
                  shape (train_time, test_time, classes, predict_shape)
        The mean prediction for each training and each testing time point
        for each class.
    """
    if y is None:
        y = gat.y_train_
    y_pred = list()
    for train in range(len(gat.y_pred_)):
        y_pred_ = list()
        for test in range(len(gat.y_pred_[train])):
            y_pred__ = list()
            for c in np.unique(y):
                m = np.mean(gat.y_pred_[train][test][y == c, :], axis=0)
                y_pred__.append(m)
            y_pred_.append(y_pred__)
        y_pred.append(y_pred_)
    return y_pred


def rescale_ypred(gat, clf=None, scorer=None, keep_sign=True):
    """"""
    if clf is None:
        clf = gat.clf
    if scorer is None:
        scorer = gat.scorer_

    y_pred_r = copy.deepcopy(gat.y_pred_)
    for t_train, y_pred_ in enumerate(gat.y_pred_):
        for t_test, y_pred__ in enumerate(y_pred_):
            for train, test in gat.cv_:
                n = len(y_pred__)
                X = np.reshape(y_pred__[:, 0], [n, 1])
                clf.fit(X[train], gat.y_train_[train])
                p = clf.predict(X[test])
                if keep_sign:
                    if scorer(gat.y_train_[train],
                              y_pred__[train].squeeze()) < .5:
                        p[test, 0] = -p[test, 0] + 1
                y_pred_r[t_train][t_test] = p
    return y_pred_r


def zscore_ypred(gat):
    """"""
    y_pred = copy.deepcopy(gat.y_pred_)
    n_T = len(gat.train_times_['slices'])
    for t_train in range(n_T):
        n_t = len(gat.test_times_['slices'][t_train])
        for t_test in range(n_t):
            p = y_pred[t_train][t_test]
            p -= np.tile(np.mean(p, axis=0), [len(p), 1])
            p /= np.tile(np.std(p, axis=0), [len(p), 1])
            y_pred[t_train][t_test] = p
    return y_pred


class GAT(GeneralizationAcrossTime):
    def __init__(self, gat):
        for key in gat.__dict__.keys():
            setattr(self, key, getattr(gat, key))

    def score(self, y=None, scorer=None):
        if scorer is not None:
            self.scorer = scorer
        return super(GAT, self).score(y=y)

    def subscore(self, sel, y=None, scorer=None, copy=True):
        """Subscores a GAT.

        Parameters
        ----------
            sel : list or array, shape (n_predictions)
            y : None | list or array, shape (n_selected_predictions,)
                If None, y set to gat.y_true_. Defaults to None.
            copy : bool
                change GAT object

        Returns
        -------
        scores
        """
        scores = subscore(self)
        if copy is True:
            self.scores_ = scores
        return scores

    def subselect_ypred(self, sel, copy=False):
        """Select subselection of y_pred_ of GAT.

        Parameters
        ----------
            sel : list or array, shape (n_predictions)
            copy : bool
                change GAT object

        Returns
        -------
            new gat
        """
        return subselect_ypred(self, sel)

    def mean_ypred(self, y=None):
        """Provides mean prediction for each category.

        Parameters
        ----------
            y : None | list or array, shape (n_predictions,)
                If None, y set to gat.y_train_. Defaults to None.

        Returns
        -------
        mean_y_pred : list of list of (float | array),
                      shape (train_time, test_time, classes, predict_shape)
            The mean prediction for each training and each testing time point
            for each class.
        """
        return mean_ypred(self, y=y)

    def rescale_ypred(self, clf=None, scorer=None, keep_sign=True):
        """"""
        return rescale_ypred(self, clf=clf, scorer=scorer, keep_sign=keep_sign)

    def zscore_ypred(self):
        """"""
        return zscore_ypred(self)


class GATs(object):
    def __init__(self, gat_list, remove_coef=True):
        if isinstance(gat_list, GeneralizationAcrossTime):
            gat_list = [gat_list]

        if remove_coef:
            gat_list_ = list()
            for gat in gat_list:
                gat.estimators_ = list()
                gat_list_.append(gat)
            gat_list = gat_list_

        self.gat_list = gat_list
        for key in gat_list[0].__dict__().key():
            setattr(self, key, getattr(gat_list[0], key))
        if hasattr(gat_list[0], 'y_pred_'):
            self._mean_y_pred_()
        if hasattr(gat_list[0], 'scores_'):
            self._mean_scores()
        return self

    def _mean_ypreds(self):
        y_pred_list = list()
        for gat in self.gat_list:
            y_pred_list.append(gat.mean_ypred())
        self.y_pred_ = copy.deepcopy(y_pred_list[0])
        for train, y_pred_train in enumerate(self.y_pred):
            for test in range(len(y_pred_train)):
                self.y_pred_[train][test] = np.mean(y_pred_list[train][test],
                                                    axis=0)
        self.y_train_ = np.unique([gat.y_train_ for gat in self.gat_list])

    def _mean_scores(self):
        self.scores_ = np.mean([gat.scores_ for gat in self.gat_list], axis=0)

    def mean_ypreds(self, y=None):
        return self.y_preds_

    def score(self, y_list=None):
        if np.ndim(y_list) < len(self.gat_list):
            y_list = [y_list for idx in range(len(self.gat_list))]

        for gat, y in zip(self.gat_list, y_list):
            gat.score(y=y)

        self.scorer = gat.scorer_
        self.gat.scores_ = np.mean([gat.scores_ for gat in self.gat_list],
                                   axis=0)
        return self.gat.scores_


def combine_y(gat_list, order=None, n_pred=None):
    """Combines multiple gat.y_pred_ & gat.y_train_ into a single gat.

    Parameters
    ----------
        gat_list : list of GeneralizationAcrossTime objects, shape (n_gat)
            The gats must have been predicted (gat.predict(epochs))
        order : None | list, shape (n_gat), optional
            Order of the prediction, to be recombined. Defaults to None.
        n_pred : None | int, optional
            Maximum number of predictions. If None, set to max(sel). Defaults
            to None.
    Returns
    -------
        cmb_gat : GeneralizationAcrossTime object
            The combined gat object"""
    import copy
    from gat.utils import GAT

    if isinstance(gat_list, GeneralizationAcrossTime):
        gat_list = [gat_list]
        order = [order]

    for gat in gat_list:
        if not isinstance(gat, GeneralizationAcrossTime):
            raise ValueError('gat must be a GeneralizationAcrossTime object')
    gat_list = [GAT(gat) for gat in gat_list]

    if order is not None:
        if len(gat_list) != len(order):
            raise ValueError('len(order) must equal len(gat_list)')
    else:
        order = [range(len(gat.y_pred_[0][0])) for gat in gat_list]
        for idx in range(1, len(order)):
            order[idx] += len(order[idx-1])

    # Identifiy trial number
    if n_pred is None:
        n_pred = np.max([np.max(sel) for sel in order]) + 1
    n_dims = np.shape(gat_list[0].y_pred_[0][0])[1]

    # Initialize combined gat
    cmb_gat = copy.deepcopy(gat_list[0])

    # Initialize y_pred
    cmb_gat.y_pred_ = list()
    cmb_gat.cv_.n = n_pred
    cmb_gat.cv_.test_folds = np.nan * np.zeros(n_pred)
    cmb_gat.cv_.y = np.nan * np.zeros(n_pred)

    for train in range(len(gat.y_pred_)):
        y_pred_ = list()
        for test in range(len(gat.y_pred_[train])):
            y_pred_.append(np.nan * np.ones((n_pred, n_dims)))
        cmb_gat.y_pred_.append(y_pred_)

    # Initialize y_train
    cmb_gat.y_train_ = np.ones((n_pred,))

    for gat, sel in zip(gat_list, order):
        cmb_gat.y_train_[sel] = gat.y_train_
        cmb_gat.cv_.test_folds[sel] = gat.cv_.test_folds
        cmb_gat.cv_.y[sel] = gat.cv_.y
        for t_train in range(len(gat.y_pred_)):
            for t_test in range(len(gat.y_pred_[t_train])):
                cmb_gat.y_pred_[t_train][t_test][sel, :] = \
                    gat.y_pred_[t_train][t_test]
    # clean
    for att in ['scores_', 'scorer_', 'y_true_']:
        if hasattr(cmb_gat, att):
            delattr(cmb_gat, att)
    return cmb_gat
