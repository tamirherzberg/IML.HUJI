from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

SIGNS = (1, -1)


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        lowest_err = 1
        best_sign = 0
        best_feat = 0
        best_thr = 0
        for j in range(X.shape[1]):
            for sign in SIGNS:
                thr, err = self._find_threshold(X[:, j], y, sign)
                if err <= lowest_err:  # err is always <= 1, so it'll update
                    best_feat = j
                    best_thr = thr
                    best_sign = sign
        self.j_ = best_feat
        self.threshold_ = best_thr
        self.sign_ = best_sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        response = []
        for val in X[:, self.j_]:
            if val < self.threshold_:
                response.append(-self.sign_)
            else:
                response.append(self.sign_)
        return np.array(response)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold (value) by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        # sort by values, keep labels accordingly
        combined = np.vstack((values, labels))
        combined.sort(axis=0)
        values = combined[0]
        labels = combined[1]

        total_weights = np.sum(abs(labels))
        cur_loss = self._weighted_loss(labels, np.ones(labels.shape[0])) * sign
        best_loss = cur_loss
        best_ind = 0
        for i in range(1, values.shape[0]):
            if np.sign(values[i - 1]) == sign:
                cur_loss += abs(labels[i - 1])
            else:
                cur_loss -= abs(labels[i - 1])
                if cur_loss < best_loss:
                    best_ind = i
                    best_loss = cur_loss
        return values[best_ind], best_loss / total_weights

    def _weighted_loss(self, labels, preds):
        prod = np.dot(labels, preds)
        ind = np.where(prod < 0)
        loss_sum = 0
        for i in ind:
            loss_sum += abs(prod[i])
        return loss_sum

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self._weighted_loss(y, self.predict(X))
