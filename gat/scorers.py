# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)


def scorer_spearman(y_true, y_pred):
    from scipy.stats import spearmanr
    rho, p = spearmanr(y_true, y_pred[:, 0])
    return rho


def scorer_auc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelBinarizer
    le = LabelBinarizer()
    y_true = le.fit_transform(y_true)
    return roc_auc_score(y_true, y_pred)
