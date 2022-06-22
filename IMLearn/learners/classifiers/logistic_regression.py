from typing import NoReturn
import numpy as np
from IMLearn import BaseEstimator
from IMLearn.desent_methods import GradientDescent, learning_rate
from IMLearn.desent_methods.modules import LogisticModule, RegularizedModule, L1, L2
from IMLearn.metrics.loss_functions import misclassification_error

INVALID_VALUE_ERROR = "Supported penalty types are: none, l1, l2"
NO_REGULARIZATION_flag = 'none'
L2_flag = 'l2'
L1_flag = 'l1'


class LogisticRegression(BaseEstimator):
    """
    Logistic Regression Classifier

    Attributes
    ----------
    solver_: GradientDescent, default=GradientDescent()
        Descent method solver to use for the logistic regression objective optimization

    penalty_: str, default="none"
        Type of regularization term to add to logistic regression objective. Supported values
        are "none", "l1", "l2"

    lam_: float, default=1
        Regularization parameter to be used in case `self.penalty_` is not "none"

    alpha_: float, default=0.5
        Threshold value by which to convert class probability to class value

    include_intercept_: bool, default=True
        Should fitted model include an intercept or not

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by linear regression. To be set in
        `LogisticRegression.fit` function.
    """

    def __init__(self,
                 include_intercept: bool = True,
                 solver: GradientDescent = GradientDescent(learning_rate.FixedLR(1e-4),
                                                           max_iter=20000),
                 penalty: str = "none",
                 lam: float = 1,
                 alpha: float = .5):
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        solver: GradientDescent, default=GradientDescent()
            Descent method solver to use for the logistic regression objective optimization

        penalty: str, default="none"
            Type of regularization term to add to logistic regression objective. Supported values
            are "none", "l1", "l2"

        lam: float, default=1
            Regularization parameter to be used in case `self.penalty_` is not "none"

        alpha: float, default=0.5
            Threshold value by which to convert class probability to class value

        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.solver_ = solver
        self.lam_ = lam
        self.penalty_ = penalty
        self.alpha_ = alpha

        if penalty not in ["none", "l1", "l2"]:
            raise ValueError("Supported penalty types are: none, l1, l2")

        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Logistic regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model using specified `self.optimizer_` passed when instantiating class and includes an intercept
        if specified by `self.include_intercept_
        """
        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]), X]  # add ones column

        d = X.shape[1]
        init_weights = np.random.normal(0, 1, d) / np.sqrt(d)  # pdf requirement

        if self.penalty_ == NO_REGULARIZATION_flag:
            self.coefs_ = self.solver_.fit(LogisticModule(init_weights), X, y)

        else:
            if self.penalty_ == L1_flag:
                reg_module = L1(init_weights)
            elif self.penalty_ == L2_flag:
                reg_module = L2(init_weights)
            else:
                raise ValueError(INVALID_VALUE_ERROR)
            module = RegularizedModule(
                LogisticModule(init_weights), reg_module, self.lam_, init_weights, self.include_intercept_)
            self.coefs_ = self.solver_.fit(module, X, y)

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
        y_prob = self.predict_proba(X)
        return (y_prob >= self.alpha_).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of samples being classified as `1` according to sigmoid(Xw)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict probability for

        Returns
        -------
        probabilities: ndarray of shape (n_samples,)
            Probability of each sample being classified as `1` according to the fitted model
        """
        def sigmoid(t): return 1 / (1 + np.exp(-t))
        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]), X]  # add ones column
        y_pred = X @ self.coefs_
        return sigmoid(y_pred)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification error

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification error
        """
        return misclassification_error(y, self.predict(X))
