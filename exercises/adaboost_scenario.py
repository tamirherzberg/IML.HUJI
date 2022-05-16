import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.templates.default = "simple_white"
pio.renderers.default = "chrome"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ab = AdaBoost(DecisionStump, n_learners)
    ab.fit(train_X, train_y)
    noiseless_train_loss = []
    noiseless_test_loss = []
    learners_num = np.arange(1, n_learners + 1)
    for t in learners_num:
        noiseless_train_loss.append(ab.partial_loss(train_X, train_y, t))
        noiseless_test_loss.append(ab.partial_loss(test_X, test_y, t))
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=learners_num, y=noiseless_train_loss,
                              mode="lines", name="Train"))
    fig1.add_trace(go.Scatter(x=learners_num, y=noiseless_test_loss,
                              mode="lines", name="Test"))
    fig1.update_layout(title=f"Train and Test Errors as Function of Number of Learners (noise = {noise})",
                       margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[f"{t} classifiers" for t in T],
                         horizontal_spacing=0.03, vertical_spacing=.08)

    for i, t in enumerate(T):
        fig2.add_traces([decision_surface(lambda X: ab.partial_predict(X, t), lims[0], lims[1], showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=test_y, line=dict(color="black", width=1)))],
                        rows=1 if i < 2 else 2,
                        cols=(i % 2) + 1)
    fig2.update_layout(title=f"Decision Boundaries Obtained By Using Various Ensemble Sizes (noise = {noise})",
                       margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig2.show()

    # Question 3: Decision surface of best performing ensemble
    best_ensemble_idx = np.argmin(noiseless_test_loss)
    fig3 = make_subplots(rows=1, cols=1)
    fig3.add_traces(
        [decision_surface(lambda X: ab.partial_predict(X, best_ensemble_idx), lims[0], lims[1], showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=test_y, line=dict(color="black", width=1)))])
    fig3.update_layout(title=f"Decision Boundary of Best Performing Ensemble\n"
                             f"Size = {best_ensemble_idx + 1},"
                             f" Accuracy = {1 - noiseless_test_loss[best_ensemble_idx]}, noise = {noise}",
                       margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig3.show()

    # Question 4: Decision surface with weighted samples
    D_norm = ab.D_ / np.max(ab.D_) * 5
    fig4 = go.Figure([decision_surface(ab.predict, lims[0], lims[1], showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=train_y, size=D_norm,
                                             line=dict(color="black", width=1)))],
                     layout=dict(title=f"Decision Surface of Weighted Training Set (noise = {noise})"))
    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    # Question 5:
    fit_and_evaluate_adaboost(noise=0.4)
