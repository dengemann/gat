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
    import copy
    gat_ = copy.deepcopy(gat)
    # Subselection of trials
    gat.y_pred_ = list()
    for train in range(len(gat_.y_pred_)):
        y_pred_ = list()
        for test in range(len(gat_.y_pred_[train])):
            y_pred_.append(gat_.y_pred_[train][test][sel, :])
        gat.y_pred_.append(y_pred_)
    # gat.y_train_ = gat.y_train_[sel]  # XXX
    gat.y_pred_ = gat.y_pred_
    if scorer is not None:
        gat.scorer = scorer
    return gat.score(y=y)


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
