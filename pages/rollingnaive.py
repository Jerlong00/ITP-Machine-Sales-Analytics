import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from llm import generate_llm_recommendation

st.set_page_config(page_title="Rolling Naive Forecast", layout="wide")
st.title("🌀 Rolling Naive Forecast (Daily / Weekly / Biweekly / Monthly)")

# ✅ Access shared file from session
df = st.session_state.get("df")

if df is not None:
    try:
        df.columns = df.columns.astype(str).str.strip()
        df['saleDate'] = pd.to_datetime(df['saleDate'], errors='coerce')
        df = df.dropna(subset=['saleDate'])

        if not all(col in df.columns for col in ['locationId', 'Qty']):
            st.error("❌ Required columns 'locationId' and 'Qty' not found.")
            st.stop()

        # Selections
        machines = df['locationId'].unique().tolist()
        selected_machine = st.selectbox("🔧 Select Machine", machines)

        freq_choice = st.selectbox("📊 Forecast Frequency", ["Daily", "Weekly", "Monthly"])
        freq_map = {
            "Daily": "D",
            "Weekly": "W",
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
        cutoff = forecast_start - pd.Timedelta(days=365)
        df_train = df_resampled[(df_resampled['ds'] >= cutoff) & (df_resampled['ds'] < forecast_start)]
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
        valid_mask = y_true != 0
        if np.any(valid_mask):
            mape = np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100
        else:
            mape = np.nan
        rolling_change = (
    df_eval['y'].replace(0, np.nan)
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .mean()
) * 100



        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("MAPE", f"{mape:.2f}%" if not np.isnan(mape) else "N/A")
        col4.metric("Rolling Change", f"{rolling_change:.2f}%")
        # Compute confidence interval for actuals after forecast starts
        train_std = df_train['y'].std()
        residual_std = (df_eval['y'] - df_eval['y_pred']).rolling(window=3, min_periods=1).std().fillna(0)
        ci_upper = df_eval['y_pred'] + 1.96 * residual_std
        ci_lower = df_eval['y_pred'] - 1.96 * residual_std

        # Chart window
        cutoff_date = pd.to_datetime("2024-12-01")
        forecast_end = df_eval['ds'].iloc[-1]
        df_plot = df_resampled[(df_resampled['ds'] >= cutoff_date) & (df_resampled['ds'] <= forecast_end)].copy()
        df_plot.loc[df_plot['ds'] < forecast_start, 'y_pred'] = np.nan

        # Forecast chart
        st.subheader("📊 Forecast Chart")
        fig, ax = plt.subplots(figsize=(12, 5))

        df_plot.set_index('ds')['y'].plot(ax=ax, label='Actual', marker='o', color='blue')
        df_plot.set_index('ds')['y_pred'].plot(ax=ax, label='Naive Forecast', linestyle='--', marker='x', color='orange')
        # Add 95% CI band starting from forecast start
        ci_plot = df_eval.set_index('ds')[['y']].copy()
        ci_plot['ci_lower'] = ci_lower.values
        ci_plot['ci_upper'] = ci_upper.values
        ci_plot = ci_plot[(ci_plot.index >= forecast_start) & (ci_plot.index <= forecast_end)]

        ax.fill_between(
            ci_plot.index,
            ci_plot['ci_lower'],
            ci_plot['ci_upper'],
            color='orange',
            alpha=0.2,
            label='95% CI (Naive Prediction)'
        )

        for x, y in zip(df_plot['ds'], df_plot['y']):
            if pd.notna(y):
                ax.text(x, y + 0.5, f"{y:.0f}", color='blue', fontsize=9, ha='center')
        for x, y_pred in zip(df_plot['ds'], df_plot['y_pred']):
            if pd.notna(y_pred):
                ax.text(x, y_pred + 0.5, f"{y_pred:.0f}", color='orange', fontsize=9, ha='center')
        

        ax.axvline(x=forecast_start, linestyle='--', color='gray', label="Forecast Start")
        ax.set_title(f"Rolling Naive Backtest ({forecast_steps} Steps) — {selected_machine}", fontsize=18)
        ax.set_ylabel("Qty", fontsize=14)
        ax.set_xlabel("Date", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.grid(True)
        
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
else:
    st.warning("⚠️ No uploaded file found in session. Please upload your data on the main page.")

# Summary for LLM
forecast_summary = {
    "machine": selected_machine,
    "trend": "increasing" if rolling_change > 0 else "decreasing" if rolling_change < 0 else "flat",
    "weekend_peak": any(df_eval['ds'].dt.weekday >= 5),
    "holiday_next_week": False,  # You can inject real logic later
    "last_week_avg_sales": df_train['y'].iloc[-7:].mean() if len(df_train) >= 7 else df_train['y'].mean()
}
# 🧠 Optional LLM-based AI Recommendations
if st.checkbox("🧠 Show AI Recommendations Based on Forecast"):
    with st.spinner("🧠 Thinking... generating suggestions..."):
        try:
            suggestions = generate_llm_recommendation(forecast_summary)
            st.subheader("💡 AI-Generated Operational Suggestions")
            st.markdown(suggestions)
        except Exception as e:
            st.error(f"❌ LLM error: {e}")


