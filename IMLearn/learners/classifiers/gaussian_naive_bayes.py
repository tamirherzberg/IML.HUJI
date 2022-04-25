from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_ = np.unique(y)
        self._set_mu_pi(X, y)
        self._set_vars(X, y)
        self.fitted_ = True

    def _set_vars(self, X, y):
        """
        calculates and sets the MLE variance. Assumes self.classe, self.mum, self.pi are already set
        """
        vars_list = []
        for k in range(len(self.classes_)):
            cur_x = X[y == self.classes_[k]]
            sum_list = []
            for i in range(len(cur_x)):
                sum_list.append(np.square(cur_x[i] - self.mu_[k]))
            sum_list = np.array(sum_list)
            vars_list.append(np.sum(sum_list, axis=0) / (len(cur_x) - 1))
        self.vars_ = np.array(vars_list)

    def _set_mu_pi(self, X: np.ndarray, y: np.ndarray):
        """
        calculates and sets the MLE Mean and Pi. Assumes self.classes is already set
        """
        mu = []
        pi_list = []
        for k in self.classes_:
            x_list = X[y == k]
            n_k = len(x_list)
            pi_list.append(n_k / X.shape[0])
            mu.append(np.sum(x_list, axis=0) / n_k)
        self.mu_ = np.array(mu)
        self.pi_ = np.array(pi_list)

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
        for k_ind in range(len(self.classes_)):
            sum_list = []
            for j in range(X.shape[1]):
                sum_list.append(np.log(np.sqrt(2 * np.pi * self.vars_[k_ind][j]))
                                + 0.5 * (((X[:, j] - self.mu_[k_ind][j]) ** 2) / self.vars_[k_ind][j]))
            sum_list = np.array(sum_list)
            lh_list.append(np.log(self.pi_[k_ind]) - np.sum(sum_list, axis=0))
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
