import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Rolling Naive Forecast", layout="wide")
st.title("🌀 Rolling Naive Forecast (Daily / Weekly / Biweekly / Monthly)")

uploaded_file = st.file_uploader("📤 Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        filename = uploaded_file.name.lower()
        df = pd.read_csv(uploaded_file) if filename.endswith(".csv") else pd.read_excel(uploaded_file)
        df.columns = df.columns.astype(str).str.strip()
        df['saleDate'] = pd.to_datetime(df['saleDate'], errors='coerce')
        df = df.dropna(subset=['saleDate'])

        if not all(col in df.columns for col in ['locationId', 'Qty']):
            st.error("❌ Required columns 'locationId' and 'Qty' not found.")
            st.stop()

        # Selections
        machines = df['locationId'].unique().tolist()
        selected_machine = st.selectbox("🔧 Select Machine", machines)

        freq_choice = st.selectbox("📊 Forecast Frequency", ["Daily", "Weekly", "Biweekly", "Monthly"])
        freq_map = {
            "Daily": "D",
            "Weekly": "W",
            "Biweekly": "2W",
            "Monthly": "M"
        }
        selected_freq = freq_map[freq_choice]
        forecast_steps = st.slider("⏱️ Forecast Steps (2025)", 2, 60, 14)

        # Preprocessing
        df_machine = df[df['locationId'] == selected_machine].copy()
        df_machine.set_index('saleDate', inplace=True)
        df_machine = df_machine.sort_index()
        df_resampled = df_machine['Qty'].resample(selected_freq).sum().reset_index()
        df_resampled.rename(columns={'saleDate': 'ds', 'Qty': 'y'}, inplace=True)

        forecast_start = pd.to_datetime("2025-01-01")
        df_train = df_resampled[(df_resampled['ds'] >= "2023-01-01") & (df_resampled['ds'] < forecast_start)].copy()
        df_2025 = df_resampled[df_resampled['ds'] >= forecast_start].copy()

        if len(df_2025) < forecast_steps:
            st.warning("⚠️ Not enough 2025 data to evaluate selected forecast_steps.")
            st.stop()

        # Rolling naive forecast
        df_resampled['y_pred'] = df_resampled['y'].shift(1)

        # Limit to forecast steps
        df_eval = df_2025.iloc[:forecast_steps].copy()
        df_eval['y_pred'] = df_resampled['y_pred'].iloc[-len(df_eval):].values

        # Metrics
        y_true = df_eval['y'].values
        y_pred = df_eval['y_pred'].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
        df_train['diff'] = df_train['y'].diff()
        rolling_mae = np.mean(np.abs(df_train['diff']))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("MAPE", f"{mape:.2f}%")
        col4.metric("Rolling MAE (Train)", f"{rolling_mae:.2f}")

        # Chart window: 1 month before + forecast steps
        cutoff_date = pd.to_datetime("2024-12-01")
        forecast_end = df_eval['ds'].iloc[-1]
        df_plot = df_resampled[
            (df_resampled['ds'] >= cutoff_date) &
            (df_resampled['ds'] <= forecast_end)
        ].copy()

        # Hide forecast before 2025
        df_plot.loc[df_plot['ds'] < forecast_start, 'y_pred'] = np.nan

        # Plot
        st.subheader("📊 Forecast Chart")
        fig, ax = plt.subplots(figsize=(10, 4))
        # Plot lines
        df_plot.set_index('ds')['y'].plot(ax=ax, label='Actual', marker='o', color='blue')
        df_plot.set_index('ds')['y_pred'].plot(ax=ax, label='Naive Forecast', linestyle='--', marker='x', color='orange')

        # Add value labels
        for x, y in zip(df_plot['ds'], df_plot['y']):
            if pd.notna(y):
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8, color='blue')

        for x, y_pred in zip(df_plot['ds'], df_plot['y_pred']):
            if pd.notna(y_pred):
                ax.annotate(f"{y_pred:.0f}", (x, y_pred), textcoords="offset points", xytext=(0, -10), ha='center', fontsize=8, color='orange')

        # Reference line and legend
        ax.axvline(x=forecast_start, linestyle='--', color='gray', label="Forecast Start")
        ax.set_title(f"Rolling Naive Backtest ({forecast_steps} Days) — {selected_machine}")
        ax.set_ylabel("Qty")
        ax.legend()
        st.pyplot(fig)


        # CSV download
        df_download = df_eval[['ds', 'y', 'y_pred']].rename(columns={'y': 'Actual', 'y_pred': 'Forecast'})
        st.download_button(
            label="📥 Download Forecast CSV",
            data=df_download.to_csv(index=False).encode(),
            file_name=f"{selected_machine}_{freq_choice.lower()}_naive_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")


