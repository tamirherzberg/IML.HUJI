from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"  # didn't show it to me otherwise
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
    go.Figure([go.Scatter(x=list(range(10, 1001, 10)), y=mean_distance, mode='markers')],
              layout=go.Layout(title="2) Samples Quantity Impact on Absolute Distance Between The Estimated And True Value Of The Expectation",
                               xaxis_title="Samples Quantity",
                               yaxis_title="Absolute Distance From Mean")).show()
    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure([go.Scatter(x=X, y=uni_g.pdf(X), mode='markers')],
              layout=go.Layout(
                  title="3) Empirical PDF Of Fitted Model",
                  xaxis_title="Samples",
                  yaxis_title="Density")).show()



def test_multivariate_gaussian():
    pass
    # # Question 4 - Draw samples and print fitted model
    # raise NotImplementedError()
    #
    # # Question 5 - Likelihood evaluation
    # raise NotImplementedError()
    #
    # # Question 6 - Maximum likelihood
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
    ######### draft : a graph that works (from lab)
    # ms = np.linspace(2, 200, 200).astype(np.int32)
    # mu, sigma = 5, 2
    # estimated_sigmas = []
    # new_estimated_sigmas = []
    # for m in ms:
    #     X = np.random.normal(mu, sigma, size=m)
    #     estimated_sigmas.append(X.var(ddof=1))
    #     new_estimated_sigmas.append(np.mean(abs(X - 5)))
    #
    # go.Figure([go.Scatter(x=ms, y=estimated_sigmas, mode='markers+lines', name=r'$\widehat\sigma^2$'),
    #            go.Scatter(x=ms, y=new_estimated_sigmas, mode='markers+lines', name=r'$\widehat\new sigma^2$'),
    #            go.Scatter(x=ms, y=[sigma ** 2] * len(ms), mode='lines', name=r'$\sigma^2$')],
    #           layout=go.Layout(title=r"$\text{(6) Estimation of Variance As Function Of Number Of Samples}$",
    #                            xaxis_title="$m\\text{ - number of samples}$",
    #                            yaxis_title="r$\hat\sigma^2$",
    #                            height=300)).show()
    #########
