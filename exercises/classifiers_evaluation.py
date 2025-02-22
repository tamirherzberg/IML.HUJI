from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

pio.renderers.default = "chrome"

SYMBOLS = np.array(['square', 'circle', 'cross'])


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def calc_loss(arg_perc, c_x, c_y):
            losses.append(arg_perc.loss(X, y))

        perc = Perceptron(callback=calc_loss)
        perc.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        go.Figure(data=go.Scatter(x=np.arange(start=1, stop=len(losses) + 1), y=np.array(losses)),
                  layout=dict(title=n + " Data - Loss Value As A Function Of Iterations Amount",
                              xaxis_title="Number of iterations",
                              yaxis_title="Loss")).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda_classifier = LDA()
        lda_classifier.fit(X, y)

        gn_classifier = GaussianNaiveBayes()
        gn_classifier.fit(X, y)

        models = [gn_classifier, lda_classifier]

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        lda_prediction = lda_classifier.predict(X)
        gn_prediction = gn_classifier.predict(X)
        predicitons = [gn_prediction, lda_prediction]

        # Create subplots
        from IMLearn.metrics import accuracy

        lda_acc = accuracy(y, lda_prediction)
        gn_acc = accuracy(y, gn_prediction)

        plots = make_subplots(rows=1, cols=2,
                              subplot_titles=[f"Gaussian Naive Bayes | Accuracy: {gn_acc}",
                                              f"LDA algorithm | Accuracy: {lda_acc}"])
        plots.update_layout(title=f"Dataset: {f}")

        # Add traces for data-points setting symbols and colors
        for i, p in enumerate(predicitons):
            plots.add_trace(row=1, col=i + 1, trace=go.Scatter(
                x=X[:, 0], y=X[:, 1],
                showlegend=False,
                mode='markers', marker=dict(color=p, symbol=SYMBOLS[y], line=dict(width=0.75, color='black'))))

            # Add `X` dots specifying fitted Gaussians' means
            mu = models[i].mu_
            plots.add_trace(row=1, col=i + 1,
                            trace=go.Scatter(x=mu[:, 0], y=mu[:, 1],
                                             showlegend=False,
                                             mode='markers', marker=dict(color='black', symbol='x',
                                                                         line=dict(width=0.5, color='black'))))

        # Add ellipses depicting the covariances of the fitted Gaussians
        for j in range(3):
            plots.add_trace(row=1, col=1,
                            trace=get_ellipse(gn_classifier.mu_[j], np.diag(gn_classifier.vars_[j])))

            plots.add_trace(row=1, col=2,
                            trace=get_ellipse(lda_classifier.mu_[j], lda_classifier.cov_))

        plots.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

    # quiz q1:

    # S = {(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 2), (7, 2)}
    # X1 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # y1 = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    # gn1 = GaussianNaiveBayes()
    # gn1.fit(X1, y1)
    # print("pi = " + str(gn1.pi_[0]))
    # print("mu = " + str(gn1.mu_[1]))

    # quiz q2:
    # S = {([1, 1], 0), ([1, 2], 0), ([2, 3], 1), ([2, 4], 1), ([3, 3], 1), ([3, 4], 1)}
    # X2 = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
    # y2 = np.array([0, 0, 1, 1, 1, 1])
    # gn2 = GaussianNaiveBayes()
    # gn2.fit(X2, y2)
    # print("sigma^2[1,0]= " + str(gn2.vars_[0][0]))  # supposed to be ?
    # print("sigma^2[1,1]=" + str(gn2.vars_[1][0]))  # supposed to be 0.33?
