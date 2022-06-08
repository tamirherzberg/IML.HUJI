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
    eps = np.random.normal(0, scale=noise, size=n_samples)
    y = f(X) + eps
    pd_y = pd.Series(y)
    pd_X = pd.DataFrame(X)
    train_X, train_y, test_X, test_y = split_train_test(pd_X, pd_y, 2 / 3)
    train_X, train_y, test_X, test_y = np.concatenate(np.array(train_X)), np.array(train_y), np.concatenate(
        np.array(test_X)), np.array(test_y)

    fig1 = go.Figure(
        [go.Scatter(x=X, y=(f(X)), mode="markers", name="Real Points",
                    marker=dict(color="black", opacity=.7)),
         go.Scatter(x=test_X, y=test_y, mode="markers", name="Test Set",
                    marker=dict(color="red", opacity=.7)),
         go.Scatter(x=train_X, y=train_y, mode="markers", name="Train Set",
                    marker=dict(color="blue", opacity=.7))
         ],
        layout=go.Layout(
            title=f"Real Model vs Noise (Sigma = {noise }) Model Train and Test Sets",
            xaxis_title="Sample",
            yaxis_title="Response")
    )
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_scores, val_scores = [], []
    ks = np.arange(11)
    for k in ks:
        avg_train_score, avg_val_score = cross_validate(
            PolynomialFitting(k), train_X, train_y, mean_square_error, cv=5)
        train_scores.append(avg_train_score)
        val_scores.append(avg_val_score)
    train_scores, val_scores = np.array(train_scores), np.array(val_scores)
    fig2 = go.Figure([
        go.Scatter(x=ks, y=train_scores, mode="lines+markers", name="Train Average Score"),
        go.Scatter(x=ks, y=val_scores, mode="lines+markers", name="Validation Average Score")],
        layout=go.Layout(
            title=f"Average Train and Validation Errors As A Function of Polyfit Degree",
            xaxis_title="Polyfit Degree",
            yaxis_title="MSE"))
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(val_scores)
    model = PolynomialFitting(best_k)
    model.fit(X, y)
    model_test_error = np.round(mean_square_error(train_y, model.predict(train_X)), 2)
    print(f"Best K value: {best_k}, Its Model's Test Error: {model_test_error}.")


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
    X_full, y_full = datasets.load_diabetes(return_X_y=True)
    n_samples_list = [n_samples]
    train_X, test_X = np.split(X_full, n_samples_list, axis=0)
    train_y, test_y = np.split(y_full, n_samples_list, axis=0)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lamda_range = np.linspace(0.01, 2, n_evaluations)
    ridge_train_scores, ridge_val_scores = [], []
    lasso_train_scores, lasso_val_scores = [], []
    for lamda in lamda_range:
        lasso_avg_train, lasso_avg_val = cross_validate(Lasso(alpha=lamda), train_X, train_y, mean_square_error)
        ridge_avg_train, ridge_avg_val = cross_validate(RidgeRegression(lam=lamda), train_X, train_y, mean_square_error)
        lasso_train_scores.append(lasso_avg_train)
        lasso_val_scores.append(lasso_avg_val)
        ridge_train_scores.append(ridge_avg_train)
        ridge_val_scores.append(ridge_avg_val)
    fig3 = go.Figure([
        go.Scatter(x=lamda_range, y=ridge_train_scores, mode="lines", name="Ridge Train Average Score"),
        go.Scatter(x=lamda_range, y=ridge_val_scores, mode="lines", name="Ridge Validation Average Score"),
        go.Scatter(x=lamda_range, y=lasso_train_scores, mode="lines", name="Lasso Train Average Score"),
        go.Scatter(x=lamda_range, y=lasso_val_scores, mode="lines", name="Lasso Validation Average Score")
    ], layout=go.Layout(
        title=f"Average Train and Validation Errors For Different Regularization Constants",
        xaxis_title="Lamda Values",
        yaxis_title="MSE")
    )
    fig3.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_optimal_lamda_val = lamda_range[np.argmin([ridge_val_scores])]
    print(f"Lamda = {ridge_optimal_lamda_val} minimizes Ridge validation error")
    lasso_optimal_lamda_val = lamda_range[np.argmin([lasso_val_scores])]
    print(f"Lamda = {lasso_optimal_lamda_val} minimizes Lasso validation error")
    opt_ridge = RidgeRegression(lam=ridge_optimal_lamda_val)
    opt_lasso = Lasso(alpha=lasso_optimal_lamda_val)
    least_squares = LinearRegression()
    opt_lasso.fit(X_full, y_full)
    opt_ridge.fit(X_full, y_full)
    least_squares.fit(X_full, y_full)
    print(f"Ridge test error {opt_ridge.loss(test_X, test_y)}")
    print(f"Lasso test error {mean_square_error(test_y, opt_lasso.predict(test_X))}")
    print(f"Least Squares test error {least_squares.loss(test_X, test_y)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
