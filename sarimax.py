# Full version: Cleaned, clipped forecast, and patched exog handling
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import zscore
import numpy as np
import warnings

# Helper: Ensure exog variables are numeric
def enforce_numeric_exog(df_resampled):
    exog_vars = ["is_weekend", "is_holiday", "is_school_holiday"]
    for col in exog_vars:
        if col not in df_resampled.columns:
            df_resampled[col] = 0
        df_resampled[col] = df_resampled[col].astype(float)
    return df_resampled[exog_vars]

# Helper: Generate exog for future
def generate_future_exog(start_date, periods, freq):
    future_index = pd.date_range(start=start_date, periods=periods, freq=freq)
    future_exog = pd.DataFrame(index=future_index)
    future_exog["is_weekend"] = (future_index.weekday >= 5).astype(float)
    future_exog["is_holiday"] = future_index.isin([
        pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-10"),
        pd.Timestamp("2023-04-07"), pd.Timestamp("2023-05-01"),
        pd.Timestamp("2023-06-02"), pd.Timestamp("2023-08-09"),
        pd.Timestamp("2023-11-11"), pd.Timestamp("2023-12-25"),
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-10"),
        pd.Timestamp("2024-04-07"), pd.Timestamp("2024-05-01"),
        pd.Timestamp("2024-06-02"), pd.Timestamp("2024-08-09"),
        pd.Timestamp("2024-11-11"), pd.Timestamp("2024-12-25")
    ]).astype(float)
    future_exog["is_school_holiday"] = future_index.month.isin([6, 12]).astype(float)
    return future_exog

# Streamlit UI
st.set_page_config(page_title="AI Sales Forecasting", layout="wide")
st.title("📈 AI Sales Forecasting App (Daily/Weekly/Bi-weekly/Monthly)")
st.markdown("✅ Upload your Excel or CSV file with machine sales data.")

uploaded_file = st.file_uploader("📤 Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        filename = uploaded_file.name.lower()
        df = pd.read_csv(uploaded_file) if filename.endswith(".csv") else pd.read_excel(uploaded_file)
        df.columns = df.columns.astype(str).str.strip().str.replace(' ', '').str.replace(':', r'\:')

        st.write("📋 Columns found:")
        st.write(df.columns.tolist())

        date_column_candidates = [col for col in df.columns if 'date' in col.lower()]
        if not date_column_candidates:
            st.warning("⚠️ No 'date' column found. Defaulting to first column.")
            date_column_candidates = [df.columns[0]]

        selected_date_col = st.selectbox("📅 Select the date column:", date_column_candidates)
        df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
        df.dropna(subset=[selected_date_col], inplace=True)
        df.set_index(selected_date_col, inplace=True)

        freq = st.selectbox("📊 Forecast Frequency:", ["Daily", "Weekly", "Bi-weekly", "Monthly"])
        freq_map = {"Daily": "D", "Weekly": "W", "Bi-weekly": "2W", "Monthly": "M"}
        seasonal_m_map = {"Daily": 365, "Weekly": 52, "Bi-weekly": 26, "Monthly": 12}
        min_obs_map = {"Daily": 90, "Weekly": 30, "Bi-weekly": 20, "Monthly": 18}
        selected_freq = freq_map[freq]
        seasonal_m = seasonal_m_map[freq]
        min_required = min_obs_map[freq]

        if "locationId" in df.columns and "Qty" in df.columns:
            df_pivoted = df.pivot_table(index=df.index, columns="locationId", values="Qty", aggfunc="sum")
            machine_cols = df_pivoted.columns.astype(str).tolist()
            selected_machine = st.selectbox("🛠 Select machine to forecast:", machine_cols)

            resample_method = "sum" if freq in ["Daily", "Weekly", "Bi-weekly"] else "mean"
            df_resampled = df_pivoted[selected_machine].resample(selected_freq).agg(resample_method).to_frame(name=selected_machine)

            df_resampled["is_weekend"] = (df_resampled.index.weekday >= 5).astype(float)
            public_holidays = pd.to_datetime([
                "2023-01-01", "2023-02-10", "2023-04-07", "2023-05-01",
                "2023-06-02", "2023-08-09", "2023-11-11", "2023-12-25",
                "2024-01-01", "2024-02-10", "2024-04-07", "2024-05-01",
                "2024-06-02", "2024-08-09", "2024-11-11", "2024-12-25"
            ])
            df_resampled["is_holiday"] = df_resampled.index.isin(public_holidays).astype(float)
            df_resampled["is_school_holiday"] = df_resampled.index.month.isin([6, 12]).astype(float)

            z_scores = zscore(df_resampled[selected_machine].bfill())
            df_resampled["cleaned"] = df_resampled[selected_machine]
            df_resampled.loc[(abs(z_scores) > 3) & (df_resampled["is_holiday"] == 0), "cleaned"] = np.nan
            df_resampled["cleaned"] = df_resampled["cleaned"].interpolate(method="linear")
            df_resampled["cleaned_filled"] = df_resampled["cleaned"].bfill().clip(lower=0.01)

            step_defaults = {"Daily": 30, "Weekly": 12, "Bi-weekly": 8, "Monthly": 6}
            forecast_steps = st.slider(
                f"📆 How many {freq.lower()} periods to forecast?",
                2, step_defaults[freq]*2, step_defaults[freq]
            )

            if len(df_resampled) < min_required:
                seasonal_order = (0, 0, 0, 0)
                st.warning("⚠️ Not enough data to model seasonality.")
            else:
                seasonal_order = (1, 1, 1, seasonal_m)

            if freq == "Daily" and len(df_resampled) > 365:
                df_resampled = df_resampled.tail(365)
                st.info("🧪 Using last 365 days of data.")

            exog = enforce_numeric_exog(df_resampled)
            future_exog = generate_future_exog(
                df_resampled.index[-1] + pd.tseries.frequencies.to_offset(selected_freq),
                forecast_steps,
                selected_freq
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = SARIMAX(
                    df_resampled["cleaned_filled"],
                    exog=exog,
                    order=(1, 1, 1),
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend='t'
                )
                results = model.fit(disp=False)

            forecast = results.get_forecast(steps=forecast_steps, exog=future_exog)
            forecast_df = forecast.predicted_mean.rename(f"{selected_machine}_forecast").clip(lower=0.01)
            forecast_df.index.freq = selected_freq

            st.subheader(f"📉 {selected_machine} Sales (Last 2 Months + Forecast - {freq})")
            display_start = df_resampled.index.max() - pd.DateOffset(days=60)
            df_display = df_resampled.loc[df_resampled.index >= display_start]

            if df_display["cleaned_filled"].dropna().empty:
                st.warning("⚠️ No recent historical data.")
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                df_display["cleaned_filled"].plot(ax=ax, label="Historical (last 2 months)")
                forecast_df.plot(ax=ax, label="Forecast", linestyle="--")
                ax.set_ylabel("Sales")
                ax.legend()
                st.pyplot(fig)

            actual = df_resampled["cleaned_filled"].iloc[-forecast_steps:]
            predicted = forecast_df
            actual = actual[-len(predicted):]
            mape = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100

            st.subheader("📊 Forecast Accuracy Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mean_absolute_error(actual, predicted):.2f}")
            col2.metric("RMSE", f"{mean_squared_error(actual, predicted) ** 0.5:.2f}")
            col3.metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "N/A")

            st.download_button(
                label="📥 Download Forecast CSV",
                data=forecast_df.to_csv().encode(),
                file_name=f"{selected_machine}_{freq.lower()}_forecast.csv",
                mime="text/csv"
            )

        else:
            st.error("❌ Your file must contain 'locationId' and 'Qty' columns.")
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
