# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np


def scorer_spearman(y_true, y_pred):
    from scipy.stats import spearmanr
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 0]
    sel = np.where(~np.isnan(y_true + y_pred))[0]
    rho, p = spearmanr(y_true[sel], y_pred[sel])
    return rho


def scorer_auc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelBinarizer
    le = LabelBinarizer()
    y_true = le.fit_transform(y_true)
    return roc_auc_score(y_true, y_pred)
