from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_, self._m = None, None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self._m = X.shape[0]
        self.classes_ = np.unique(y)
        self._set_mu_pi(X, y)
        self._set_covs(X, y)
        self.fitted_ = True

    def _set_mu_pi(self, X: np.ndarray, y: np.ndarray):
        """
        calculates and sets the MLE Mean and Pi. Assumes self.classes is already set
        """
        mu = []
        pi_list = []
        for k in self.classes_:
            x_list = X[y == k]
            n_k = len(x_list)
            pi_list.append(n_k / self._m)
            mu.append(np.sum(x_list, axis=0) / n_k)
        self.mu_ = np.array(mu)
        self.pi = np.array(pi_list)

    def _set_covs(self, X: np.ndarray, y: np.ndarray):
        """
        calculates and sets the MLE cov and cov inv matrices
        """
        sum_list = []
        for i in range(self._m):
            k = np.where(self.classes_ == y[i])
            v = X[i] - self.mu_[k]
            sum_list.append(np.outer(v, v))
        self.cov_ = np.sum(sum_list, axis=0) / (self._m - self.classes_.shape[0])
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        k_ind = np.argmax(self.likelihood(X), axis=1)
        return self.classes_[k_ind]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        lh_list = []
        for k in range(len(self.classes_)):
            a_k = self._cov_inv @ self.mu_[k]
            b_k = np.log(self.pi[k]) - 0.5 * self.mu_[k] @ self._cov_inv @ self.mu_[k]
            lh_list.append(a_k @ X.T + b_k)
        lh_list = np.array(lh_list)
        return lh_list.T

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
