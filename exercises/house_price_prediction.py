from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

FILENAME = "../datasets/house_prices.csv"

CENTER_LAT = 47.62

CENTER_LONG = -122.24

pio.templates.default = "simple_white"
pio.renderers.default = "chrome"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) as a single
    DataFrame
    """
    # load and delete missing values, duplicates
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    preprocess(full_data)
    return full_data


def preprocess(full_data):
    full_data.drop(columns=['id', 'zipcode', 'date'], inplace=True)
    full_data.drop(full_data.index[full_data['bedrooms'] == 33], inplace=True)
    full_data.drop(full_data.index[full_data['bathrooms'] <= 0], inplace=True)
    full_data.drop(full_data.index[full_data['sqft_living'] > 13000], inplace=True)  # removes a fishy entry
    full_data.drop(full_data.index[full_data['price'] <= 0], inplace=True)
    # handle yr_renovated
    update_est_year_built(full_data)
    update_distance(full_data)


def update_est_year_built(df: pd.DataFrame):
    """
    Estimate year built, taking into consideration if it renovated and if so when.
    Subtracts a maximum of 50 years from the renovation year, depending on
    how big the difference between the house building and renovation.
    The calculation was based on the maximum span between the two years
    in the dataset, which is 130.
    """
    for i, sample in df.iterrows():
        if df.loc[i, 'yr_renovated'] != 0:
            df.loc[i, 'yr_built'] = df.loc[i, 'yr_renovated'] \
                                    - int(((df.loc[i, 'yr_renovated'] - df.loc[i, 'yr_built']) / 130) * 65)
            # option B: just change year to renovated date
            # df.loc[i, 'yr_built'] = df.loc[i, 'yr_renovated']
    df.rename(columns={'yr_built': 'yr_est'}, inplace=True)
    df.drop(columns='yr_renovated', inplace=True)


def update_distance(df: pd.DataFrame):
    for i, sample in df.iterrows():
        df.loc[i, 'lat'] = dist_from_center(df.loc[i, 'lat'], df.loc[i, 'long'])
    df.rename(columns={'lat': 'dist_from_center'}, inplace=True)
    df.drop(columns='long', inplace=True)

def dist_from_center(lat2, lon2):
    """
    Function from gpxpy package, copied to avoid dependecy.
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    # >>> origin = (48.1372, 11.5756)  # Munich
    # >>> destination = (52.5186, 13.4083)  # Berlin
    # >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = CENTER_LAT, CENTER_LONG
    radius = 6371  # earth radius in km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return -d

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feat in X.columns:
        cor = pearson_cor(X[feat], y)
        go.Figure([go.Scatter(x=X[feat], y=y, mode='markers')],
                  layout=go.Layout(
                      title=f"Price as a function of {feat}<br>"
                            f"<sup>Pearson Correlation: {cor}</sup>",
                      xaxis_title=feat,
                      yaxis_title="Price")).write_image(output_path + f"/price_{feat}_graph.png")


def pearson_cor(X: pd.Series, Y: pd.Series):
    """
    Calculates the Pearson Correlation for X,Y being one of the features and a
    response vec, accordingly
    """
    return Y.cov(X) / (np.std(X) * np.std(Y))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df = load_data(FILENAME)

    # Question 2 - Feature evaluation with respect to response
    y = df['price']
    X = df.drop(columns=['price'])
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    p_values = np.arange(10, 101)
    means = []
    stds = []
    train_X.insert(0, 'price', train_y)

    for p in p_values:
        p_loss = []
        for i in range(10):
            cur_sam = train_X.sample(frac=p / 100)
            cur_X = cur_sam.drop(columns='price')
            cur_y = cur_sam['price']
            estimator = LinearRegression()
            estimator.fit(cur_X, cur_y)
            p_loss.append(estimator.loss(cur_X, cur_y))
        p_loss = np.array(p_loss)
        means.append(p_loss.mean())
        stds.append(p_loss.std())

    means = np.array(means)
    stds = np.array(stds)

    # creates graph
    go.Figure(
        [go.Scatter(x=p_values, y=means - 2 * stds, fill=None, mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=p_values, y=means + 2 * stds, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=p_values, y=means, mode="markers+lines", marker=dict(color="black", size=1), showlegend=False)],
        layout=go.Layout(title=r"MSE Loss As A Function Of Training Size",
                         xaxis_title="Percentage Of Training Set",
                         yaxis_title="MSE",
                         height=300)).show()
