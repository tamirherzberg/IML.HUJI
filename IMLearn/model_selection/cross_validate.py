from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X_cpy: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    X_cpy = deepcopy(X)
    folds_idx_list = [i * (X_cpy.shape[0] // cv) for i in range(cv)] + [X_cpy.shape[0]]
    train_errors = []
    val_errors = []
    for i in range(cv - 1):
        fold_idx_range = np.arange(folds_idx_list[i], folds_idx_list[i + 1])
        cur_val_X, cur_val_y = X_cpy[fold_idx_range], y[fold_idx_range]
        cur_train_X, cur_train_y = np.delete(X_cpy, fold_idx_range, 0), np.delete(y, fold_idx_range, 0)
        estimator = BaseEstimator()
        estimator.fit(cur_train_X, cur_train_y)
        cur_train_err = scoring(cur_train_X, cur_train_y)
        cur_val_err = scoring(cur_val_X, cur_val_y)
        train_errors.append(cur_train_err)
        val_errors.append(cur_val_err)
    return np.array(train_errors).mean(), np.array(val_errors).mean()
