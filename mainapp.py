import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
from prophet import Prophet
=======
from sklearn.metrics import mean_absolute_error, mean_squared_error
>>>>>>> Stashed changes
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import zscore, boxcox
from scipy.special import inv_boxcox
<<<<<<< Updated upstream
import numpy as np
import os
import warnings

from prophet.plot import plot_components

warnings.filterwarnings("ignore")
=======
from prophet import Prophet
>>>>>>> Stashed changes

# Streamlit setup
st.set_page_config(page_title="AI Sales Forecasting", layout="wide")
st.title("📈 AI Sales Forecasting App (Multi-Frequency)")

# File Upload
uploaded_file = st.file_uploader("📤 Upload Excel or CSV File", type=["xlsx", "xls", "csv"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = df.columns.astype(str).str.strip().str.replace("\xa0", "").str.replace(":", "")

        # Date Handling
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        selected_date_col = date_cols[0] if date_cols else df.columns[0]
        df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
        df = df.dropna(subset=[selected_date_col]).set_index(selected_date_col)

        # Filter outliers
        if "outlier_flag" in df.columns:
            df = df[df["outlier_flag"] == 0]

        # Resample frequency selection
        freq_dict = {
            "Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Bi-annually": "2Q", "Yearly": "Y"
        }
        freq_label = st.selectbox("📊 Forecast Frequency:", list(freq_dict.keys()))
        selected_freq = freq_dict[freq_label]

        # Machine & Resampling
        if "locationId" in df.columns and "Qty" in df.columns:
            df_pivot = df.pivot_table(index=df.index, columns="locationId", values="Qty", aggfunc="sum")
            selected_machine = st.selectbox("🛠 Select machine:", df_pivot.columns)
            resample_method = "sum" if selected_freq in ["D", "W"] else "mean"
            df_resampled = df_pivot[selected_machine].resample(selected_freq).agg(resample_method).to_frame("Qty")
        else:
            st.error("Your data must include 'locationId' and 'Qty'")
            st.stop()

<<<<<<< Updated upstream
        use_boxcox = st.checkbox("📐 Apply Box-Cox transformation (stabilize variance)", value=False)

        z_scores = zscore(df_resampled[selected_machine].fillna(method="bfill"))
        threshold = 3
        df_resampled["cleaned"] = df_resampled[selected_machine]
        df_resampled.loc[(z_scores > threshold) | (z_scores < -threshold), "cleaned"] = np.nan
        df_resampled["cleaned"].interpolate(method="linear", inplace=True)
        df_resampled["cleaned_filled"] = df_resampled["cleaned"].fillna(method="bfill").clip(lower=0.01)
=======
        # Box-Cox option
        use_boxcox = st.checkbox("📐 Use Box-Cox transformation", value=False)
        cleaned = df_resampled["Qty"].copy()
        cleaned[zscore(cleaned.fillna(method="bfill")) > 3] = np.nan
        cleaned = cleaned.interpolate().fillna(method="bfill").clip(lower=1)
>>>>>>> Stashed changes

        if use_boxcox:
            cleaned, lam = boxcox(cleaned)
        else:
            lam = None

        # Model selection
        model_type = st.radio("🤖 Select model:", ["SARIMAX", "Prophet"], horizontal=True)

        # Forecast steps
        step_defaults = {"D": 30, "W": 12, "M": 12, "Q": 8, "2Q": 4, "Y": 3}
        forecast_steps = st.slider(
            f"📆 Forecast how many {freq_label.lower()}s?",
            2, step_defaults[selected_freq] * 2, step_defaults[selected_freq]
        )

<<<<<<< Updated upstream
        # ✅ Model selection
        model_choice = st.radio("🧠 Select Forecasting Model:", ["SARIMAX", "Prophet"])

        seasonal_periods = {"D": 7, "W": 52, "M": 12, "Q": 4, "2Q": 2, "Y": 1}
        seasonal_m = seasonal_periods[selected_freq]
=======
        train = cleaned[:-forecast_steps]
        test = df_resampled["Qty"][-forecast_steps:]
>>>>>>> Stashed changes

        if model_type == "Prophet":
            st.subheader("📈 Prophet Forecast with Confidence Interval")

<<<<<<< Updated upstream
        # ✅ Forecast logic
        if model_choice == "SARIMAX":
            model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,seasonal_m))
            results = model.fit(disp=False)
            predicted = results.forecast(steps=forecast_steps)

            if use_boxcox:
                predicted = inv_boxcox(predicted, lam)

            predicted = pd.Series(predicted, index=test.index)

        else:
            prophet_df = df_resampled.reset_index()[[selected_date_col, selected_machine]].rename(columns={
                selected_date_col: "ds",
                selected_machine: "y"
            })

            if use_boxcox:
                prophet_df["y"] = boxcox(prophet_df["y"].replace(0, 0.01), lmbda=lam)[0]

            m = Prophet()
            m.fit(prophet_df)

            future = m.make_future_dataframe(periods=forecast_steps, freq=selected_freq)
            forecast = m.predict(future)

            predicted = forecast.set_index("ds")["yhat"].iloc[-forecast_steps:]

            if use_boxcox:
                predicted = inv_boxcox(predicted, lam)

            prophet_model = m
            prophet_forecast = forecast

        # ✅ Forecast Accuracy Metrics
        actual = test
        non_zero_actual = actual.replace(0, np.nan)
        rmse = mean_squared_error(actual, predicted) ** 0.5
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / non_zero_actual)) * 100
        rolling_mae = np.mean(np.abs(df_resampled[selected_machine].diff()))
=======
            df_prophet = pd.DataFrame({
                "ds": df_resampled.index,
                "y": df_resampled["Qty"]
            })

            model = Prophet(
                changepoint_prior_scale=0.3,
                interval_width=0.95,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=forecast_steps, freq=selected_freq)
            forecast = model.predict(future)
>>>>>>> Stashed changes

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_prophet["ds"], df_prophet["y"], label="Historical", color="black", marker='o', markersize=3, linewidth=1)
            ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="orange", linestyle="--", marker='x', markersize=4)
            ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="orange", alpha=0.2, label="95% Confidence Interval")
            ax.set_ylabel("Sales")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)

<<<<<<< Updated upstream
        # ✅ Forecast Plot
        st.subheader("📈 Forecast vs Historical")
        fig2, ax2 = plt.subplots()
        df_resampled[selected_machine].plot(ax=ax2, label="Historical")
        pd.Series(predicted, index=test.index).plot(ax=ax2, label="Forecast", linestyle="--")
        ax2.set_ylabel("Sales")
        ax2.legend()
        st.pyplot(fig2)

        # ✅ Prophet component breakdown
        if model_choice == "Prophet":
            st.subheader("🧠 Prophet Component Breakdown")
            fig3 = plot_components(prophet_model, prophet_forecast)
            st.pyplot(fig3)

        # ✅ Download forecast
        st.download_button(
            label="📥 Download Forecast CSV",
            data=pd.Series(predicted, index=test.index).to_csv().encode(),
            file_name=f"{selected_machine}_{freq.lower()}_{model_choice.lower()}_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
=======
            # Accuracy
            df_actual = test.resample("7D").sum()
            df_forecast = forecast.set_index("ds")["yhat"].iloc[-forecast_steps:].resample("7D").sum()
            common_index = df_actual.index.intersection(df_forecast.index)

            mae = mean_absolute_error(df_actual.loc[common_index], df_forecast.loc[common_index])
            rmse = mean_squared_error(df_actual.loc[common_index], df_forecast.loc[common_index], squared=False)
            mape = np.mean(np.abs((df_actual - df_forecast) / df_actual.replace(0, np.nan))) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("MAPE", f"{mape:.2f}%")

        elif model_type == "SARIMAX":
            seasonal_map = {"D": 7, "W": 52, "M": 12, "Q": 4, "2Q": 2, "Y": 1}
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_map[selected_freq]))
            results = model.fit(disp=False)
            forecast = results.forecast(steps=forecast_steps)

            if use_boxcox:
                forecast = inv_boxcox(forecast, lam)

            fig2, ax2 = plt.subplots()
            df_resampled["Qty"].plot(ax=ax2, label="Historical")
            pd.Series(forecast, index=test.index).plot(ax=ax2, label="SARIMAX Forecast", linestyle="--")
            ax2.set_ylabel("Sales")
            ax2.legend()
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Error: {e}")
>>>>>>> Stashed changes
