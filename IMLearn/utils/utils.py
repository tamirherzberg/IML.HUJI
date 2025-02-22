from typing import Tuple
import numpy as np
import pandas as pd
from copy import deepcopy


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X_cpy : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    X_cpy = deepcopy(X)
    X_cpy.insert(0, 'response', y)  # to make sure we don't mix up samples' response
    shuffled = X_cpy.sample(frac=1)
    comb_train = shuffled.iloc[:int(np.ceil(train_proportion * len(shuffled)))]
    comb_test = shuffled.iloc[int(np.ceil(train_proportion * len(shuffled))):]
    train_y = comb_train['response']
    test_y = comb_test['response']
    train_X = comb_train.drop(columns='response')
    test_X = comb_test.drop(columns='response')
    return train_X, train_y, test_X, test_y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    a_unique_vals = sorted(list(set(a)))  # sort unique values only
    b_unique_vals = sorted(list(set(b)))
    a_dict = {val: ind for ind, val in enumerate(a_unique_vals)}
    b_dict = {val: ind for ind, val in enumerate(b_unique_vals)}
    conf_mat = np.zeros([len(a_unique_vals), len(b_unique_vals)])
    for k in range(a.shape[0]):
        i = a_dict[a[k]]
        j = b_dict[b[k]]
        conf_mat[i][j] += 1
    return conf_mat
