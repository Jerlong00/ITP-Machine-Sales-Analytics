import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import zscore
import warnings
from llm import generate_llm_recommendation

st.set_page_config(page_title="AI Sales Forecasting", layout="wide")
st.title("📈 AI Sales Forecasting App (SARIMAX)")

# ✅ Use shared dataframe from session
df = st.session_state.get("df")

if df is not None:
    df = df.copy()
    selected_date_col = "saleDate"

    df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
    df.dropna(subset=[selected_date_col], inplace=True)
    df.set_index(selected_date_col, inplace=True)

    freq = st.selectbox("📊 Forecast Frequency:", ["Daily", "Weekly", "Monthly"])
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
    seasonal_m_map = {"Daily": 30, "Weekly": 8, "Monthly": 3}
    max_steps_map = {"Daily": 90, "Weekly": 12, "Monthly": 3}
    selected_freq = freq_map[freq]
    seasonal_m = seasonal_m_map[freq]
    max_forecast_steps = max_steps_map[freq]

    forecast_steps = st.slider("⏱️ Forecast Steps (Future Periods)", 1, max_forecast_steps, min(7, max_forecast_steps))

    if "locationId" in df.columns and "Qty" in df.columns:
        machines = sorted(df["locationId"].unique().tolist())
        selected_machine = st.selectbox("🛠 Select machine to forecast:", machines, index=0)

        df_machine = df[df["locationId"] == selected_machine].copy()
        df_machine = df_machine.sort_index()
        df_resampled = df_machine["Qty"].resample(selected_freq).sum().to_frame(name="Qty")

        df_resampled["is_weekend"] = (df_resampled.index.weekday >= 5).astype(float)
        public_holidays = pd.to_datetime([
            "2023-01-01", "2023-02-10", "2023-04-07", "2023-05-01",
            "2023-06-02", "2023-08-09", "2023-11-11", "2023-12-25",
            "2024-01-01", "2024-02-10", "2024-04-07", "2024-05-01",
            "2024-06-02", "2024-08-09", "2024-11-11", "2024-12-25",
            "2025-01-01", "2025-01-29", "2025-03-31", "2025-04-18"
        ])
        df_resampled["is_holiday"] = df_resampled.index.isin(public_holidays).astype(float)
        df_resampled["is_school_holiday"] = df_resampled.index.month.isin([6, 12]).astype(float)

        z_scores = zscore(df_resampled["Qty"].bfill())
        df_resampled["cleaned"] = df_resampled["Qty"]
        df_resampled.loc[(abs(z_scores) > 3) & (df_resampled["is_holiday"] == 0), "cleaned"] = np.nan
        df_resampled["cleaned"] = df_resampled["cleaned"].interpolate(method="linear")
        df_resampled["cleaned_filled"] = df_resampled["cleaned"].bfill().clip(lower=0.01)

        if freq == "Daily" and len(df_resampled) > 365:
            df_resampled = df_resampled.tail(365)

        unit_label = {"D": "day", "W": "week", "M": "month"}[selected_freq]
        max_history = len(df_resampled)
        default_show = min(30, max_history)

        history_window = st.slider(
            f"📂 Display last N {unit_label}s of history",
            min_value=2,
            max_value=max_history,
            value=default_show
        )
        df_display = df_resampled.tail(history_window)

        if forecast_steps >= len(df_display):
            st.warning(f"⚠️ Not enough history to compare {forecast_steps} steps. Showing last {len(df_display)//2} steps instead.")
            forecast_steps = len(df_display) // 2

        exog = df_resampled[["is_weekend", "is_holiday", "is_school_holiday"]]
        future_index = pd.date_range(
            start=df_resampled.index[-1] + pd.tseries.frequencies.to_offset(selected_freq),
            periods=forecast_steps,
            freq=selected_freq
        )
        future_exog = pd.DataFrame(index=future_index)
        future_exog["is_weekend"] = (future_index.weekday >= 5).astype(float)
        future_exog["is_holiday"] = future_index.isin(public_holidays).astype(float)
        future_exog["is_school_holiday"] = future_index.month.isin([6, 12]).astype(float)

        with st.spinner("🔄 Training SARIMAX model... please wait"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = SARIMAX(
                    df_resampled["cleaned_filled"],
                    exog=exog,
                    order=(1, 0, 0),
                    seasonal_order=(1, 1, 1, seasonal_m),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)

        forecast = results.get_forecast(steps=forecast_steps, exog=future_exog)
        forecast_df = forecast.predicted_mean.rename("Forecast").clip(lower=0.01)
        forecast_df.index.freq = selected_freq

        if not forecast_df.isna().all():
            train_data = df_display["cleaned_filled"].iloc[:-forecast_steps].copy()
            test_data = df_display["cleaned_filled"].iloc[-forecast_steps:].copy()
            test_pred_obj = results.get_prediction(
                start=test_data.index[0],
                end=test_data.index[-1],
                exog=exog[-forecast_steps:]
            )
            test_predictions = test_pred_obj.predicted_mean
            test_ci = test_pred_obj.conf_int(alpha=0.05)  # 95% CI

            
            mae = mean_absolute_error(test_data, test_predictions)
            rmse = mean_squared_error(test_data, test_predictions) ** 0.5
            mape = np.mean(np.abs((test_data - test_predictions) / test_data.replace(0, np.nan))) * 100
            rolling_change = test_data.pct_change().mean() * 100

            st.subheader("📊 SARIMAX Accuracy Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{mae:.2f}")
            c2.metric("RMSE", f"{rmse:.2f}")
            c3.metric("MAPE", f"{mape:.2f}%")
            c4.metric("Rolling Change", f"{rolling_change:.2f}%")

            forecast_summary = {
    "machine": selected_machine,
    "trend": "increasing" if rolling_change > 0 else "decreasing" if rolling_change < 0 else "flat",
    "weekend_peak": exog["is_weekend"].tail(7).mean() > 0.5,
    "holiday_next_week": exog["is_holiday"].tail(7).mean() > 0.3,
    "last_week_avg_sales": df_resampled["cleaned_filled"].iloc[-7:].mean()
}


        st.subheader(f"📉 {selected_machine} Sales Forecast ({freq} - Next {forecast_steps} period(s))")
        fig, ax = plt.subplots(figsize=(12, 5))
        train_data.plot(ax=ax, label="Historical Training Data", color="blue", marker='o')
        test_data.plot(ax=ax, label="Validation Actuals", color="purple", marker='o')

        if not test_predictions.isna().all():
            test_predictions.plot(ax=ax, label="Model Predictions (Test)", color="orange", linestyle="--", marker='x')
        if not forecast_df.isna().all():
            forecast_df.plot(ax=ax, label="Model Forecast (Future)", color="green", linestyle="--", marker='x')
        # Plot 95% Confidence Interval for validation predictions
        if 'test_ci' in locals():
            ax.fill_between(
                test_ci.index,
                test_ci.iloc[:, 0],  # lower bound
                test_ci.iloc[:, 1],  # upper bound
                color='orange',
                alpha=0.2,
                label='95% CI (Validation)'
            )
        for x, y in train_data.items():
            ax.text(x, y + 0.5, f"{y:.0f}", color='blue', fontsize=9, ha='center')
        for x, y in test_data.items():
            ax.text(x, y + 0.5, f"{y:.0f}", color='purple', fontsize=9, ha='center')
        for x, y in test_predictions.items():
            ax.text(x, y + 0.5, f"{y:.0f}", color='orange', fontsize=9, ha='center')
        for x, y in forecast_df.items():
            ax.text(x, y + 0.5, f"{y:.0f}", color='green', fontsize=9, ha='center')

        ax.set_title(f"{selected_machine} Sales Forecast - {freq}", fontsize=18)
        ax.set_ylabel("Sales", fontsize=14)
        ax.set_xlabel("Date", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.grid(True)

        st.pyplot(fig)

        from llm import generate_llm_recommendation

        if st.checkbox("🧠 Show AI Recommendations Based on Forecast"):
            with st.spinner("🧠 Thinking... generating suggestions..."):
                try:
                    suggestions = generate_llm_recommendation(forecast_summary)
                    st.subheader("💡 AI-Generated Operational Suggestions")
                    st.markdown(suggestions)
                except Exception as e:
                    st.error(f"❌ LLM error: {e}")

        st.download_button(
            label="📅 Download Forecast CSV",
            data=forecast_df.to_csv().encode(),
            file_name=f"{selected_machine}_{freq.lower()}_sarimax_forecast.csv",
            mime="text/csv"
        )
    else:
        st.error("❌ Your file must contain 'locationId' and 'Qty' columns.")
else:
    st.warning("⚠️ Please upload a file in the Home tab.")
