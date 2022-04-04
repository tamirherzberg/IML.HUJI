from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # load and delete missing values, duplicates
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    preprocess(full_data)


def preprocess(full_data):
    # delete 33 rooms house, 0 bathrooms houses
    full_data.drop(columns=['id', 'zipcode', 'date'], inplace=True)
    full_data.drop(full_data.index[full_data['bedrooms'] == 33], inplace=True)
    full_data.drop(full_data.index[full_data['bathrooms'] <= 0], inplace=True)
    full_data.drop(full_data.index[full_data['sqft_living'] > 13000], inplace=True)  # removes a fishy entry
    # handle yr_renovated
    update_est_year_built(full_data)
    # handle categorical data?


def update_est_year_built(df: pd.DataFrame):
    """
    Estimate year built, taking into consideration if it renovated and if so when.
    Subtracts a maximum of 50 years from the renovation year, depending on
    how big the difference between the house building and renovation.
    The calculation was based on the maximum span between the two years
    in the dataset, which is 130.
    """
    for i, val in enumerate(df['yr_built']):
        if df['yr_renovated'][i] != 0:
            df['yr_built'][i] -= ((df['yr_renovated'][i] - val) / 130) * 50
    df.rename(columns={'yr_built': 'yr_est'})


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
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    raise NotImplementedError()

    # Question 2 - Feature evaluation with respect to response
    raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
