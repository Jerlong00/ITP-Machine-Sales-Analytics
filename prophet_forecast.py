from prophet import Prophet
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.special import inv_boxcox

# Public holidays
public_holidays = pd.to_datetime([
    "2023-01-01", "2023-02-10", "2023-04-07", "2023-05-01",
    "2023-06-02", "2023-08-09", "2023-11-11", "2023-12-25",
    "2024-01-01", "2024-02-10", "2024-04-07", "2024-05-01",
    "2024-06-02", "2024-08-09", "2024-11-11", "2024-12-25"
])

def run_prophet_forecast(
    df_resampled,
    series_to_use,
    forecast_steps,
    selected_freq,
    selected_machine,
    freq
):
    df_prophet = df_resampled.copy()
    df_prophet = df_prophet[[selected_machine]].rename(columns={selected_machine: "y"})
    df_prophet["y"] = df_prophet["y"].rolling(3, min_periods=1).mean()
    df_prophet["ds"] = df_prophet.index

    # Add regressors
    df_prophet["is_weekend"] = (df_prophet["ds"].dt.weekday >= 5).astype(float)
    df_prophet["is_holiday"] = df_prophet["ds"].isin(public_holidays).astype(float)
    df_prophet["is_school_holiday"] = df_prophet["ds"].dt.month.isin([6, 12]).astype(float)

    y_cap = df_prophet["y"].max() * 1.5
    df_prophet["cap"] = y_cap
    df_prophet["floor"] = 0

    model = Prophet(
        growth="linear",
        changepoint_prior_scale=0.01,
        daily_seasonality=(freq == "Daily"),
        weekly_seasonality=(freq in ["Daily", "Weekly"]),
        yearly_seasonality=True,
        interval_width=0.9
    )
    model.add_regressor("is_weekend")
    model.add_regressor("is_holiday")
    model.add_regressor("is_school_holiday")
    model.fit(df_prophet[["ds", "y", "cap", "floor", "is_weekend", "is_holiday", "is_school_holiday"]])

    future = model.make_future_dataframe(periods=forecast_steps, freq=selected_freq)
    future["cap"] = y_cap
    future["floor"] = 0
    future["is_weekend"] = (future["ds"].dt.weekday >= 5).astype(float)
    future["is_holiday"] = future["ds"].isin(public_holidays).astype(float)
    future["is_school_holiday"] = future["ds"].dt.month.isin([6, 12]).astype(float)

    forecast = model.predict(future)
    forecast.set_index("ds", inplace=True)
    forecast = forecast[["yhat", "yhat_lower", "yhat_upper"]].clip(lower=0, upper=y_cap)
    forecast_series = forecast["yhat"].rename(f"{selected_machine}_forecast")

    validation_map = {"D": 10, "W": 20, "M": 6}
    validation_steps = validation_map.get(selected_freq, min(10, forecast_steps))
    test_actual = df_resampled[selected_machine].iloc[-validation_steps:]
    train_actual = df_resampled[selected_machine].iloc[-(validation_steps + 60):-validation_steps]

    # Split forecast
    model_prediction = forecast_series.loc[test_actual.index]
    model_future = forecast_series[forecast_series.index > test_actual.index[-1]]

    test_pred = model_prediction
    mae = mean_absolute_error(test_actual, test_pred)
    rmse = mean_squared_error(test_actual, test_pred) ** 0.5
    mape = np.mean(np.abs((test_actual - test_pred) / test_actual.replace(0, np.nan))) * 100

    st.subheader("📊 Forecast Accuracy Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "N/A")
    st.caption(f"⚠️ Forecast clipped to 0 - {y_cap:.1f} range for stability.")

    st.subheader("🔎 Display Settings")
    max_periods = len(df_resampled)
    display_periods = st.slider(
        f"📆 Show last N {freq.lower()} periods of historical data + forecast",
        min_value=7,
        max_value=max_periods,
        value=min(60, max_periods)
    )
    train_actual = df_resampled[selected_machine].iloc[-(validation_steps + display_periods):-validation_steps]

    # Plotting
    st.subheader(f"📈 {selected_machine} Sales Forecast - {freq}")
    fig, ax = plt.subplots(figsize=(11, 5))

    # 1. Training data
    ax.plot(train_actual.index, train_actual.values, color="blue", marker="o", label="Historical Training Data")
    for x, y in zip(train_actual.index, train_actual.values):
        ax.text(x, y, f"{y:.0f}", fontsize=8, ha='center', va='bottom', color="blue")

    # 2. Validation actuals
    ax.plot(test_actual.index, test_actual.values, color="purple", marker="o", label="Validation Actuals")
    for x, y in zip(test_actual.index, test_actual.values):
        ax.text(x, y, f"{y:.0f}", fontsize=8, ha='center', va='bottom', color="purple")

    # 3. Predictions (test)
    ax.plot(model_prediction.index, model_prediction.values, color="orange", linestyle="--", marker="x", label="Model Predictions (Test)")
    for x, y in zip(model_prediction.index, model_prediction.values):
        ax.text(x, y, f"{y:.0f}", fontsize=8, ha='center', va='bottom', color="orange")

    # 4. Forecast (future)
    ax.plot(model_future.index, model_future.values, color="green", linestyle="--", marker="x", label="Model Forecast (Future)")
    for x, y in zip(model_future.index, model_future.values):
        ax.text(x, y, f"{y:.0f}", fontsize=8, ha='center', va='bottom', color="green")

    # Styling
    ax.set_title(f"{selected_machine} Sales Forecast - {freq}", fontsize=18)
    ax.set_ylabel("Sales", fontsize=14)
    ax.set_xlabel("Date", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    # Date format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    # Legend outside right
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    st.pyplot(fig)

    # Download
    st.download_button(
        label="📥 Download Prophet Forecast CSV",
        data=forecast_series.to_csv().encode(),
        file_name=f"{selected_machine}_{freq.lower()}_prophet_forecast.csv",
        mime="text/csv"
    )


