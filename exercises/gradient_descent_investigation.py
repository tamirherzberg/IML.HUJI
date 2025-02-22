import numpy as np
import pandas as pd
import plotly.io as pio
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error

import plotly.graph_objects as go

pio.renderers.default = "chrome"  # didn't show it to me otherwise
pio.templates.default = "simple_white"

NO_REGULARIZATION_flag = 'none'
L2_flag = 'l2'
L1_flag = 'l1'


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(**kwargs):
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for objective in [L1, L2]:
        conv_rate_plot = go.Figure()
        obj_min_loss = np.inf
        for eta in etas:
            conv_rate_plot.update_layout(title=f"Gradient Descent Convergence Rate<br>"
                             f"<sup>{objective.__name__} Objective</sup>")
            f = objective(init.copy())
            state_callback, vals, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=state_callback)
            min_solution = gd.fit(f, X=np.empty(0), y=np.empty(0))
            des_path_title = f"{objective.__name__} Objective Descent Path<br>" \
                             f"<sup>Eta = {eta}</sup>"
            plot_descent_path(objective, np.array(weights), des_path_title).show()
            conv_rate_plot.add_trace(
                go.Scatter(x=np.arange(len(vals)), y=np.array(vals), mode='markers+lines', name=f"eta = {eta}"))
            eta_min_val = min(vals)
            if eta_min_val < obj_min_loss:
                obj_min_loss = eta_min_val
        conv_rate_plot.show()
        print(f"Minimum loss achieved in {objective.__name__} module is {obj_min_loss}")  # TODO: make sure


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    conv_rate_plot = go.Figure().update_layout(
        title="L1 Objective Gradient Descent Convergence Rate For Different Gammas"
    )
    lowest_norm = np.inf
    desc_plot = None
    for gamma in gammas:
        f = L1(init.copy())
        state_callback, vals, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=ExponentialLR(eta, gamma),
                             callback=state_callback)
        min_solution = gd.fit(f, X=np.empty(0), y=np.empty(0))
        conv_rate_plot.add_trace(go.Scatter(
            x=np.arange(len(vals)), y=np.array(vals),
            name=f'gamma = {gamma}', mode='markers+lines'
        ))
        if gamma == .95:
            des_path_title = f"{L1.__name__} Objective Descent Path<br>" \
                             f"<sup>eta = {eta}, gamma = {gamma}</sup>"
            desc_plot = plot_descent_path(L1, np.array(weights), des_path_title)
        min_gamma_norm = min(vals)
        if min_gamma_norm < lowest_norm:
            lowest_norm = min_gamma_norm
    # Plot algorithm's convergence for the different values of gamma
    conv_rate_plot.show()
    print(f"Minimum norm achieved using exponential decay is {lowest_norm}")
    # Plot descent path for gamma=0.95
    desc_plot.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
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
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    from sklearn.metrics import roc_curve, auc
    log_reg = LogisticRegression(
        solver=GradientDescent(FixedLR(1e-4), max_iter=20000))
    log_reg.fit(np.array(X_train), np.array(y_train))
    y_prob = log_reg.predict_proba(np.array(X_train))
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    alpha_star = thresholds[np.argmax(tpr - fpr)]
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()
    print(f"alpha_star = {np.round(alpha_star, 2)}")
    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    for penalty in [L1_flag, L2_flag]:
        train_score_list, val_score_list = [], []
        lamdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        for lamda in lamdas:
            train_score, val_score = cross_validate(
                LogisticRegression(
                    solver=GradientDescent(FixedLR(1e-4), max_iter=20000),
                    penalty=penalty, lam=lamda),
                np.array(X_train), np.array(y_train), misclassification_error)
            train_score_list.append(train_score)
            val_score_list.append(val_score)
        opt_lam_idx = np.argmin(val_score_list)
        opt_lamda = lamdas[opt_lam_idx]
        opt_log_reg = LogisticRegression(
            solver=GradientDescent(FixedLR(1e-4), max_iter=20000),
            penalty=penalty, lam=opt_lamda)
        opt_log_reg.fit(np.array(X_train), np.array(y_train))
        model_test_error = opt_log_reg.loss(np.array(X_test), np.array(y_test))
        print(f"{penalty}: {opt_lamda} was selected as optimal lamda value,"
              f" model's test error is {model_test_error}.")

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
