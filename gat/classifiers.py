# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import warnings
import scipy.sparse as sp
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV


class force_predict(object):
    def __init__(self, clf, mode='predict_proba', axis=0):
        self._mode = mode
        self._axis = axis
        self._clf = clf

    def fit(self, X, y, **kwargs):
        return self._clf.fit(X, y, **kwargs)

    def predict(self, X):
        if self._mode == 'predict_proba':
            return self._clf.predict_proba(X)[:, self._axis]
        elif self._mode == 'decision_function':
            distances = self._clf.decision_function(X)
            if len(distances.shape) > 1:
                return distances[:, self._axis]
            else:
                return distances
        else:
            return self._clf.predict(X)

    def get_params(self, deep=True):
        return dict(clf=self._clf, mode=self._mode, axis=self._axis)


class force_weight(object):
    def __init__(self, clf, weights=None):
        self._clf = clf

    def fit(self, X, y):
        return self._clf.fit(X, y[:, 0], sample_weight=y[:, 1])

    def predict(self, X):
        return self._clf.predict(X)

    def get_params(self, deep=True):
        return dict(clf=self._clf)


def LinearSVC_Proba(probability=False, method='sigmoid', cv=5, **kwargs):
    if probability is True:
        base_estimator = LinearSVC(**kwargs)
        return CalibratedClassifierCV(base_estimator=base_estimator,
                                      method=method, cv=cv)
    else:
        return LinearSVC(**kwargs)


def SVC_Light(probability=False, method='sigmoid', cv=5, **kwargs):
    """
    Similar to SVC(kernel='linear') without having to store 'support_vectors_'
     and '_dual_coef_'.
    Uses CalibrationClassifierCV if probability=True.
    """
    if probability is True:
        base_estimator = _SVC_Light(probability=True, **kwargs)
        return _SVC_Light_Proba(base_estimator=base_estimator, method=method,
                                cv=cv)
    else:
        return _SVC_Light(**kwargs)


class _SVC_Light_Proba(CalibratedClassifierCV):

    def decision_function(self, X):
        warnings.warn(
            "With 'probability=True' decision_function=predict_proba")
        return self.predict_proba(X)

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            # XXX
            raise ValueError('_SVC_Light currently does not support '
                             'probability=True for more than 2 classes.')
        super(_SVC_Light_Proba, self).fit(X, y)


class _SVC_Light(SVC):
    """
    Similar to SVC(kernel='linear') without having to store 'support_vectors_'
     and '_dual_coef_'
    """

    def __init__(self, kernel='linear', probability=False, **kwargs):
        if 'kernel' in kwargs.keys():
            raise ValueError('SVC_Light is only available when using a '
                             'linear kernel.')
        if 'probability' in kwargs.keys():
            raise RuntimeError('Currently, SVC_Light does not support '
                               'probability=True')
        super(_SVC_Light, self).__init__(kernel=kernel,
                                         probability=probability, **kwargs)

    def fit(self, X, y, scaling=None):
        super(_SVC_Light, self).fit(X, y)
        # compute coef from support vectors once only
        self._coef_ = self._compute_coef_()
        self.__delattr__('support_vectors_')
        self.__delattr__('_dual_coef_')

    def _compute_coef_(self):
        # Originally coef_(self) from SVC
        coef = self._get_coef()
        if sp.issparse(coef):
            coef.data.flags.writeable = False
        else:
            coef.flags.writeable = False
        return coef

    def predict(self, X):
        from gat.predicters import predict_OneVsOne
        distances = self.decision_function(X)
        y_pred = predict_OneVsOne(distances, self.classes_)
        return y_pred

    def decision_function(self, X):
        X = self._validate_for_predict(X)
        n_sample = X.shape[0]
        intercept = np.tile(self.intercept_, (n_sample, 1))
        distances = np.dot(self.coef_, X.T).T + intercept
        if len(self.classes_) == 2:
            distances *= -1
        return distances

    @property
    def coef_(self):
        return self._coef_
