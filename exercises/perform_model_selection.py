from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "chrome"  # didn't show it to me otherwise
pio.templates.default = "simple_white"


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    def f(x): return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    X = np.linspace(-1.2, 2, num=n_samples)
    eps = np.random.normal(scale=noise, size=n_samples)
    y = f(X) + eps
    pd_y = pd.Series(y)
    pd_X = pd.DataFrame(X)
    train_X, train_y, test_X, test_y = split_train_test(pd_X, pd_y, 2 / 3)
    train_X, train_y, test_X, test_y = np.asarray(train_X), np.asarray(train_y), np.asarray(test_X), np.asarray(test_y)

    fig1 = go.Figure([go.Scatter(x=X, y=(f(X)), mode="markers", name="Real Points", marker=dict(color="black", opacity=.7),
                                showlegend=False),
                     go.Scatter(x=np.concatenate(test_X), y=(test_y), mode="markers", name="Test Set",
                                marker=dict(color="red", opacity=.7),
                                showlegend=False),
                     go.Scatter(x=np.concatenate(train_X), y=(test_y), mode="markers", name="Train Set",
                                marker=dict(color="blue", opacity=.7),
                                showlegend=False)])
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    raise NotImplementedError()
