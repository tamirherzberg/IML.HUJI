from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import scipy.stats as sct

pio.renderers.default = "chrome"  # didn't show it to me otherwise
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    uni_g = UnivariateGaussian()
    X = np.random.normal(10, 1, 1000)
    uni_g.fit(X)
    print(f"({uni_g.mu_}, {uni_g.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    mean_distance = []
    for i in range(10, 1001, 10):
        uni_g.fit(X[:i])
        mean_distance.append(np.abs(uni_g.mu_ - 10))
    go.Figure([go.Scatter(x=list(range(10, 1001, 10)), y=mean_distance, mode='markers+lines')],
              layout=go.Layout(
                  title="2) Samples Quantity Impact on Absolute Distance Between The Estimated And True Value Of The Expectation",
                  xaxis_title="Samples Quantity",
                  yaxis_title="Absolute Distance From Mean")).show()
    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure([go.Scatter(x=X, y=uni_g.pdf(X), mode='markers')],
              layout=go.Layout(
                  title="3) Empirical PDF Of Fitted Model",
                  xaxis_title="Samples",
                  yaxis_title="Probability Density")).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    true_mu = np.array([0, 0, 4, 0])
    true_cov = np.array([[1, 0.2, 0, 0.5],
                         [0.2, 2, 0, 0],
                         [0, 0, 1, 0],
                         [0.5, 0, 0, 1]])
    mv_g = MultivariateGaussian()
    Y = np.random.multivariate_normal(true_mu, true_cov, 1000)
    mv_g.fit(Y)
    print(f"{mv_g.mu_}\n{mv_g.cov_}")

    # Question 5 - Likelihood evaluation
    lh_values = []
    rng = np.linspace(-10, 10, 200)  # f1,f3 values range
    # computes log-likelihood of every f1,f3 coordinate
    for f1 in rng:
        row = []
        for f3 in rng:
            alt_mu = np.array([f1, 0, f3, 0])
            row.append(mv_g.log_likelihood(alt_mu, true_cov, Y))
        lh_values.append(row)
    # generates heatmap
    go.Figure(go.Heatmap(x=rng, y=rng, z=lh_values), layout=go.Layout(
        title="5) Log-likelihood As A Function Of f1, f3 Values",
        xaxis_title="f3 value",
        yaxis_title="f1 value"
    )).show()

    # Question 6 - Maximum likelihood
    optimal_f1, optimal_f3 = np.unravel_index(np.argmax(lh_values), (200, 200))
    print(f"{rng[optimal_f3]}, {rng[optimal_f1]}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

    # q4 in quiz + tests

    # data = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #                  -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])

    # uni = UnivariateGaussian()
    # uni.fit(data)
    # print(uni.log_likelihood(1, 1, data))
    # print(uni.log_likelihood(10, 1, data))
    # print(np.sum(sct.norm.logpdf(data, 1, 1)))
    # print(np.sum(sct.norm.logpdf(data, 10, 1)))
