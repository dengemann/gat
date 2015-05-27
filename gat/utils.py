import numpy as np


def subscore(gat, sel, y=None, scorer=None):
    """Subscores a GAT.

    Parameters
    ----------
        gat : GeneralizationAcrossTime object
        sel : list or array, shape (n_predictions)
        y : None | list or array, shape (n_selected_predictions,)
            If None, y set to gat.y_true_. Defaults to None.

    Returns
    -------
    scores
    """
    gat = subpred(gat, sel)
    if scorer is not None:
        gat.scorer = scorer
    gat.score(y=y)
    return gat


def subpred(gat, sel):
    """Select subselection of y_pred_ of GAT.

    Parameters
    ----------
        gat : GeneralizationAcrossTime object
        sel : list or array, shape (n_predictions)

    Returns
    -------
    gat
    """
    import copy
    gat_ = copy.deepcopy(gat)
    # Subselection of trials
    for train in range(len(gat.y_pred_)):
        for test in range(len(gat.y_pred_[train])):
            gat_.y_pred_[train][test] = gat_.y_pred_[train][test][sel, :]
    gat_.y_train_ = gat_.y_train_[sel]
    return gat_


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
    from mne.decoding.time_gen import GeneralizationAcrossTime as GAT
    if isinstance(gat_list, GAT):
        gat_list = [gat_list]
        order = [order]

    for gat in gat_list:
        if not isinstance(gat, GAT):
            raise ValueError('gat must be a GeneralizationAcrossTime object')

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


def mean_y_pred(gat, y=None):
    """Provides mean prediction for each category.

    Parameters
    ----------
        gat : GeneralizationAcrossTime object
        y : None | list or array, shape (n_predictions,)
            If None, y set to gat.y_train_. Defaults to None.

    Returns
    -------
    mean_y_pred : list of list of (float | array),
                  shape (train_time, test_time, classes, predict_shape)
        The mean prediction for each training and each testing time point for
        each class.
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


def rescale(gat, clf=None, scorer=None, keep_sign=True):
    if clf is None:
        clf = gat.clf
    if scorer is None:
        scorer = gat.scorer_
    cv = gat.cv_
    y = gat.y_train_
    y_train = gat.y_train_
    y_pred = gat.y_pred_

    n_T = len(gat.train_times_['slices'])
    p = [list() for idx in range(n_T)]
    for t_train in range(n_T):
        n_t = len(gat.test_times_['slices'][t_train])
        p[t_train] = [list() for idx in range(n_t)]
        for t_test in range(n_t):
            p[t_train][t_test] = np.zeros(y_pred[t_train][t_test].shape)
            for train, test in cv:
                n = len(y_pred[t_train][t_test])
                X = np.reshape(y_pred[t_train][t_test][:, 0], [n, 1])
                clf.fit(X[train], y[train])
                p[t_train][t_test][test, 0] = clf.predict(X[test])
                if keep_sign:
                    if scorer(y_train[train],
                              y_pred[t_train][t_test][train].squeeze()) < .5:
                        p[t_train][t_test][test, 0] *= -1
                        p[t_train][t_test][test, 0] += 1

    gat.y_pred_ = p
    return gat


def zscore(gat, clf=None, scorer=None, keep_sign=True):
    y_pred = gat.y_pred_
    n_T = len(gat.train_times_['slices'])
    for t_train in range(n_T):
        n_t = len(gat.test_times_['slices'][t_train])
        for t_test in range(n_t):
            p = y_pred[t_train][t_test]
            p -= np.tile(np.mean(p, axis=0), [len(p), 1])
            p /= np.tile(np.std(p, axis=0), [len(p), 1])
            gat.y_pred_[t_train][t_test] = p
    return gat
