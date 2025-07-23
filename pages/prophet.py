import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.special import inv_boxcox
from scipy.stats import zscore, boxcox

st.set_page_config(page_title="PROPHET Forecast", layout="wide")
st.title("🔍 PROPHET Forecasting")

# ✅ Load shared data from session state
if "df" not in st.session_state:
    st.warning("⚠️ Please upload a file from the main page first.")
    st.stop()

df = st.session_state["df"].copy()

# Public holidays
public_holidays = pd.to_datetime([
    "2023-01-01", "2023-02-10", "2023-04-07", "2023-05-01",
    "2023-06-02", "2023-08-09", "2023-11-11", "2023-12-25",
    "2024-01-01", "2024-02-10", "2024-04-07", "2024-05-01",
    "2024-06-02", "2024-08-09", "2024-11-11", "2024-12-25"
])

if df.index.name is not None:
    df.reset_index(inplace=True)

date_cols = [c for c in df.columns if "date" in c.lower()]
if not date_cols:
    st.warning("No date column found, defaulting to first column.")
    date_cols = [df.columns[0]]

sel_date = st.selectbox("Select the date column:", date_cols)
df[sel_date] = pd.to_datetime(df[sel_date], errors="coerce")
df = df.dropna(subset=[sel_date])

if "locationId" not in df.columns or "Qty" not in df.columns:
    st.error("Your file needs 'locationId' & 'Qty' columns.")
    st.stop()

df = df.set_index(sel_date)

if "outlier_flag" in df.columns:
    df = df[df["outlier_flag"] == 0]

freq = st.selectbox("Forecast Frequency:", ["Daily", "Weekly", "Monthly"])
freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
sel_freq = freq_map[freq]

df = df[df["Qty"] > 0]
piv = df.pivot_table(index=df.index, columns="locationId", values="Qty", aggfunc="sum")
machines = piv.columns.astype(str).tolist()
sel_mach = st.selectbox("Select machine to forecast:", machines)
resamp = "sum" if freq in ["Daily", "Weekly"] else "mean"
df_res = piv[sel_mach].resample(sel_freq).agg(resamp).to_frame(name=sel_mach)
cutoff = df_res.index.max() - pd.Timedelta(days=365)
df_res = df_res[df_res.index >= cutoff]

use_bc = st.checkbox("Apply Box-Cox transformation", value=False)
zs = zscore(df_res[sel_mach].fillna(method="bfill"))
df_res["clean"] = df_res[sel_mach]
df_res.loc[(zs > 3) | (zs < -3), "clean"] = np.nan
df_res["clean"].interpolate(inplace=True)
df_res["clean_filled"] = df_res["clean"].bfill().clip(lower=0.01)

if use_bc:
    df_res["trans"], lam = boxcox(df_res["clean_filled"])
    series_to_use = df_res["trans"]
else:
    series_to_use = df_res["clean_filled"]

defaults = {"Daily": 30, "Weekly": 12, "Monthly": 12}
h = st.slider(f"How many {freq.lower()} periods to forecast?", 2, defaults[freq] * 2, defaults[freq])

df_prophet = df_res.copy()
df_prophet = df_prophet[[sel_mach]].rename(columns={sel_mach: "y"})
df_prophet["y"] = df_prophet["y"].rolling(3, min_periods=1).mean()
df_prophet["ds"] = df_prophet.index

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

future = model.make_future_dataframe(periods=h, freq=sel_freq)
future["cap"] = y_cap
future["floor"] = 0
future["is_weekend"] = (future["ds"].dt.weekday >= 5).astype(float)
future["is_holiday"] = future["ds"].isin(public_holidays).astype(float)
future["is_school_holiday"] = future["ds"].dt.month.isin([6, 12]).astype(float)

forecast = model.predict(future)
forecast.set_index("ds", inplace=True)
forecast = forecast[["yhat", "yhat_lower", "yhat_upper"]].clip(lower=0, upper=y_cap)
forecast_series = forecast["yhat"].rename(f"{sel_mach}_forecast")

validation_map = {"D": 10, "W": 20, "M": 6}
validation_steps = validation_map.get(sel_freq, min(10, h))
test_actual = df_res[sel_mach].iloc[-validation_steps:]
train_actual = df_res[sel_mach].iloc[-(validation_steps + 60):-validation_steps]

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

st.subheader("🔎 Display Settings")
max_periods = len(df_res)
display_periods = st.slider(
    f"📆 Show last N {freq.lower()} periods of historical data + forecast",
    min_value=7,
    max_value=max_periods,
    value=min(60, max_periods)
)
train_actual = df_res[sel_mach].iloc[-(validation_steps + display_periods):-validation_steps]

# 📈 Final Forecast Plot — matches x-axis layout exactly like Rolling Naive Forecast
st.subheader(f"📈 {sel_mach} Sales Forecast - {freq}")
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(train_actual.index, train_actual.values, color="blue", marker="o", label="Historical Training Data")
for x, y in zip(train_actual.index, train_actual.values):
    ax.text(x, y + 0.5, f"{y:.0f}", fontsize=8, ha='center', color="blue")

ax.plot(test_actual.index, test_actual.values, color="purple", marker="o", label="Validation Actuals")
for x, y in zip(test_actual.index, test_actual.values):
    ax.text(x, y + 0.5, f"{y:.0f}", fontsize=8, ha='center', color="purple")

ax.plot(model_prediction.index, model_prediction.values, color="orange", linestyle="--", marker="x", label="Model Predictions (Test)")
for x, y in zip(model_prediction.index, model_prediction.values):
    ax.text(x, y + 0.5, f"{y:.0f}", fontsize=8, ha='center', color="orange")

ax.plot(model_future.index, model_future.values, color="green", linestyle="--", marker="x", label="Model Forecast (Future)")
for x, y in zip(model_future.index, model_future.values):
    ax.text(x, y + 0.5, f"{y:.0f}", fontsize=8, ha='center', color="green")


# ✅ Match rnforecast x-axis layout with major month+year and minor day ticks
# ✅ Cleaner x-axis for large time ranges
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=14))  # fewer day labels
ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b\n%Y'))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))

ax.tick_params(axis='x', which='major', labelsize=10, pad=15)  # push month labels down
ax.tick_params(axis='x', which='minor', labelsize=9, rotation=0)  # horizontal days
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Sales", fontsize=14)
ax.set_title(f"{sel_mach} Sales Forecast - {freq}", fontsize=18)
ax.grid(True)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

st.pyplot(fig)

st.download_button(
    label="📥 Download Prophet Forecast CSV",
    data=forecast_series.to_csv().encode(),
    file_name=f"{sel_mach}_{freq.lower()}_prophet_forecast.csv",
    mime="text/csv"
)
