import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

FILENAME = "../datasets/City_Temperature.csv"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True).dropna().drop_duplicates()
    # preprocessing
    full_data.drop(full_data.index[full_data['Temp'] <= -70])
    full_data['DayOfYear'] = full_data['Date'].dt.day_of_year
    return full_data



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    load_data(FILENAME)

    # Question 2 - Exploring data for specific country
    # raise NotImplementedError()
    #
    # # Question 3 - Exploring differences between countries
    # raise NotImplementedError()
    #
    # # Question 4 - Fitting model for different values of `k`
    # raise NotImplementedError()
    #
    # # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()
