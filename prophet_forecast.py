from prophet import Prophet
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_prophet_forecast(df, forecast_steps, use_boxcox=False, lam=None):
    # Prepare DataFrame for Prophet
    df_prophet = df.reset_index()
    df_prophet.columns = ["ds", "y"]

    # Drop missing or zero values (Prophet is sensitive)
    df_prophet = df_prophet.dropna()
    df_prophet = df_prophet[df_prophet["y"] > 0]

    if len(df_prophet) < 2:
        raise ValueError("Not enough valid data points after cleaning for Prophet.")

    if use_boxcox:
        df_prophet["y"], lam = boxcox(df_prophet["y"])

    # Initialize Prophet with better changepoint flexibility
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.5  # More flexibility
    )
    model.add_seasonality(name='weekly', period=7, fourier_order=3)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=6)

    model.fit(df_prophet)

    # Make future dataframe using inferred frequency from index
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq is None:
        inferred_freq = "D"  # Default to daily if undetectable

    future = model.make_future_dataframe(periods=forecast_steps, freq=inferred_freq)
    forecast = model.predict(future)

    # Extract forecast and confidence interval
    forecast_output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")
    yhat = forecast_output["yhat"][-forecast_steps:]
    yhat_lower = forecast_output["yhat_lower"][-forecast_steps:]
    yhat_upper = forecast_output["yhat_upper"][-forecast_steps:]

    if use_boxcox:
        yhat = inv_boxcox(yhat, lam)
        yhat_lower = inv_boxcox(yhat_lower, lam)
        yhat_upper = inv_boxcox(yhat_upper, lam)

    return yhat, yhat_lower, yhat_upper
