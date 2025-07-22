import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import zscore, boxcox
from scipy.special import inv_boxcox
import numpy as np
import os

#✅ Outlier smoothing

#✅ Box-Cox toggle

#✅ Seasonal period logic

#✅ Forecast accuracy metrics

#✅ Rolling MAE

#✅ Smart frequency resampling

st.set_page_config(page_title="AI Sales Forecasting", layout="wide")
st.title("📈 AI Sales Forecasting App (Multi-Frequency)")

st.markdown("✅ Upload your Excel or CSV file with machine sales data. Public holiday support coming soon!")

uploaded_file = st.file_uploader("📤 Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            xls = pd.ExcelFile(uploaded_file)
            df = pd.read_excel(xls, sheet_name=0)

        df.columns = df.columns.astype(str).str.strip().str.replace('\xa0', '').str.replace(':', r'\:')

        st.write("📋 Columns found in uploaded file:")
        st.write(df.columns.tolist())

        date_column_candidates = [col for col in df.columns if 'date' in col.lower()]
        if not date_column_candidates:
            st.warning("⚠️ Couldn’t find a 'date' column — defaulting to first column")
            date_column_candidates = [df.columns[0]]

        selected_date_col = st.selectbox("📅 Select the date column:", date_column_candidates)
        df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
        df = df.dropna(subset=[selected_date_col])
        df.set_index(selected_date_col, inplace=True)

        if "outlier_flag" in df.columns:
            df = df[df["outlier_flag"] == 0]

        freq = st.selectbox("📊 Forecast Frequency:", [
            "Daily", "Weekly", "Monthly", "Quarterly", "Bi-annually", "Yearly"
        ])

        freq_map = {
            "Daily": "D",
            "Weekly": "W",
            "Monthly": "M",
            "Quarterly": "Q",
            "Bi-annually": "2Q",
            "Yearly": "Y"
        }

        selected_freq = freq_map[freq]

        if "locationId" in df.columns and "Qty" in df.columns:
            df_pivoted = df.pivot_table(index=df.index, columns="locationId", values="Qty", aggfunc="sum")
            machine_cols = df_pivoted.columns.astype(str).tolist()
            selected_machine = st.selectbox("🛠 Select machine to forecast:", machine_cols)

            resample_method = "sum" if freq in ["Daily", "Weekly"] else "mean"
            df_resampled = df_pivoted[selected_machine].resample(selected_freq).agg(resample_method).to_frame(name=selected_machine)
        else:
            st.error("❌ Your file must contain 'locationId' and 'Qty' columns.")
            st.stop()

        # ✅ Outlier smoothing and Box-Cox toggle
        use_boxcox = st.checkbox("📐 Apply Box-Cox transformation (stabilize variance)", value=False)

        z_scores = zscore(df_resampled[selected_machine].fillna(method="bfill"))
        threshold = 3
        df_resampled["cleaned"] = df_resampled[selected_machine]
        df_resampled.loc[(z_scores > threshold) | (z_scores < -threshold), "cleaned"] = np.nan
        df_resampled["cleaned"].interpolate(method="linear", inplace=True)
        df_resampled["cleaned_filled"] = df_resampled["cleaned"].fillna(method="bfill").clip(lower=0.01)

        if use_boxcox:
            df_resampled["transformed"], lam = boxcox(df_resampled["cleaned_filled"])
            series_to_use = df_resampled["transformed"]
        else:
            series_to_use = df_resampled["cleaned_filled"]

        step_defaults = {
            "Daily": 30, "Weekly": 12, "Monthly": 12,
            "Quarterly": 8, "Bi-annually": 4, "Yearly": 3
        }

        forecast_steps = st.slider(
            f"📆 How many {freq.lower()} periods to forecast?",
            2, step_defaults[freq]*2, step_defaults[freq]
        )

        seasonal_periods = {"D": 7, "W": 52, "M": 12, "Q": 4, "2Q": 2, "Y": 1}
        seasonal_m = seasonal_periods[selected_freq]

        train = series_to_use.iloc[:-forecast_steps]
        test = df_resampled[selected_machine].iloc[-forecast_steps:]

        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,seasonal_m))
        results = model.fit(disp=False)

        predicted = results.forecast(steps=forecast_steps)
        if use_boxcox:
            predicted = inv_boxcox(predicted, lam)

        actual = test
        rmse = mean_squared_error(actual, predicted) ** 0.5
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100
        rolling_mae = np.mean(np.abs(df_resampled[selected_machine].diff()))

        st.subheader("📊 Forecast Accuracy Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("MAPE", f"{mape:.2f}%")
        col4.metric("Rolling MAE", f"{rolling_mae:.2f}")

        st.subheader("📈 Forecast vs Historical")
        fig2, ax2 = plt.subplots()
        df_resampled[selected_machine].plot(ax=ax2, label="Historical")
        pd.Series(predicted, index=test.index).plot(ax=ax2, label="Forecast", linestyle="--")
        ax2.set_ylabel("Sales")
        ax2.legend()
        st.pyplot(fig2)

        st.download_button(
            label="📥 Download Forecast CSV",
            data=pd.Series(predicted, index=test.index).to_csv().encode(),
            file_name=f"{selected_machine}_{freq.lower()}_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

