import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"
pio.renderers.default = "chrome"

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
    full_data.drop(full_data.index[full_data['Temp'] <= -70], inplace=True)
    full_data['DayOfYear'] = full_data['Date'].dt.day_of_year
    return full_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(FILENAME)

    # Question 2 - Exploring data for specific country
    isr_subset = df[df['Country'] == "Israel"]
    isr_subset['Year'] = isr_subset['Year'].astype(str)
    px.scatter(isr_subset, x='DayOfYear', y='Temp', color='Year',
               title=f"Temperature in Israel As A Function Of Day Of The Year").show()

    months_std = isr_subset.groupby('Month').agg({"Temp": np.std})
    px.bar(months_std, y='Temp', barmode="group", title="Daily Temperature Standard Deviation As A Function Of Months",
    labels={"Temp": "Standard Deviation"}).show()

    # go.Figure([go.Histogram(x=isr_subset.groupby('Month'), y=months_std, showlegend=False)]).show()


    # Question 3 - Exploring differences between countries
    # raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
