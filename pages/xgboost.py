import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import zscore, boxcox
from scipy.special import inv_boxcox
import numpy as np
import os
import xgboost as xgb
from llm import generate_llm_recommendation


def plot_xgb_forecast_chart(
    df_train_plot, df_test_plot, df_last_week_plot, df_pred, selected_machine, selected_freq
):
    fig, ax = plt.subplots(figsize=(12, 5))

    # Training actuals
    if not df_train_plot.empty:
        y_vals = df_train_plot.set_index('ds')['y']
        y_vals.plot(ax=ax, label='Historical Training Data', color='blue', marker='o')
        for x, y in y_vals.items():
            ax.text(x, y + 0.5, f'{y:.0f}', color='blue', fontsize=9, ha='center')

    # Test actuals
    if not df_test_plot.empty:
        y_vals = df_test_plot.set_index('ds')['y']
        y_vals.plot(ax=ax, label='Validation Actuals', color='purple', marker='o')
        for x, y in y_vals.items():
            ax.text(x, y + 0.5, f'{y:.0f}', color='purple', fontsize=9, ha='center')

    # Test predictions with CI
    if not df_last_week_plot.empty:
        y_vals = df_last_week_plot['Predicted']
        ci_lower = df_last_week_plot['ci_lower']
        ci_upper = df_last_week_plot['ci_upper']
        
        # Plot 95% confidence band
        ax.fill_between(df_last_week_plot.index, ci_lower, ci_upper, color='orange', alpha=0.2, label='95% CI (Prediction)')
        
        # Plot prediction line
        y_vals.plot(ax=ax, label='Model Predictions (Test)', linestyle='--', marker='x', color='orange')
    
    for x, y in y_vals.items():
        ax.text(x, y + 0.5, f'{y:.0f}', color='orange', fontsize=9, ha='center')

        for x, y in y_vals.items():
            ax.text(x, y + 0.5, f'{y:.0f}', color='orange', fontsize=9, ha='center')

    # Future forecast
    if not df_pred.empty:
        y_vals = df_pred['Forecast']
        y_vals.plot(ax=ax, label='Model Forecast (Future)', linestyle='--', marker='x', color='green')
        for x, y in y_vals.items():
            ax.text(x, y + 0.5, f'{y:.0f}', color='green', fontsize=9, ha='center')

    ax.set_title(f"{selected_machine} Sales Forecast - {selected_freq}", fontsize=18)
    ax.set_ylabel("Sales", fontsize=14)
    ax.set_xlabel("Date", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(True)

    return fig

# Setup
st.set_page_config(page_title="AI Sales Forecasting", layout="wide")
st.title("📈 AI Sales Forecasting App (XGBoost Model)")

# Get shared file
df = st.session_state.get("df")

if df is not None:
    try:
        df.columns = df.columns.astype(str).str.strip().str.replace('\xa0', '').str.replace(':', ':')

        # --- Parse & clean date ---
        df['saleDate'] = pd.to_datetime(df['saleDate'], errors='coerce')
        df = df.dropna(subset=['saleDate'])

        # --- Static holidays list (can later externalize) ---
        public_holidays = pd.to_datetime([
            "2023-01-01", "2023-01-02", "2023-01-22", "2023-01-23", "2023-01-24",
            "2023-04-07", "2023-04-22", "2023-05-01", "2023-06-02", "2023-06-29",
            "2023-08-09", "2023-09-01", "2023-11-12", "2023-11-13", "2023-12-25",
            "2024-01-01", "2024-02-10", "2024-02-11", "2024-02-12", "2024-03-29",
            "2024-04-10", "2024-05-01", "2024-05-22", "2024-06-17", "2024-08-09",
            "2024-10-31", "2024-12-25",
            "2025-01-01", "2025-01-29", "2025-01-30", "2025-01-31", "2025-03-31",
            "2025-04-18", "2025-05-01", "2025-05-03", "2025-05-05", "2025-05-12",
            "2025-08-09", "2025-08-11", "2025-10-20", "2025-12-25"
        ])

        if 'locationId' in df.columns and 'Qty' in df.columns:
            machines = df['locationId'].unique().tolist()
            selected_machine = st.selectbox("🔧 Select machine to forecast:", machines)

            freq = st.selectbox("📊 Forecast Frequency:", ["Daily", "Weekly", "Monthly"])
            freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
            selected_freq = freq_map[freq]

            forecast_steps = st.slider("⏱️ Forecast Steps (Future Periods)", 2, 60, 14)

            # --- Filter for selected machine & resample ---
            df_machine = df[df['locationId'] == selected_machine].copy()
            df_machine.set_index('saleDate', inplace=True)
            df_machine = df_machine.sort_index()

            # --- Filter for selected machine & resample ---
            df_machine = df[df['locationId'] == selected_machine].copy()
            df_machine.set_index('saleDate', inplace=True)
            df_machine = df_machine.sort_index()

            df_resampled = df_machine['Qty'].resample(selected_freq).sum().reset_index()
            df_resampled.rename(columns={'saleDate': 'ds', 'Qty': 'y'}, inplace=True)

            # ✅ Trim to last 365 days of data
            cutoff = df_resampled['ds'].max() - pd.Timedelta(days=365)
            df_resampled = df_resampled[df_resampled['ds'] >= cutoff]

            # --- Feature engineering ---
            df_resampled['day_of_week'] = df_resampled['ds'].dt.dayofweek
            df_resampled['is_weekend'] = df_resampled['day_of_week'].isin([5, 6]).astype(int)
            df_resampled['month'] = df_resampled['ds'].dt.month
            df_resampled['is_holiday'] = df_resampled['ds'].isin(public_holidays).astype(int)

            # Lags
            df_resampled['lag_1'] = df_resampled['y'].shift(1)
            df_resampled['lag_2'] = df_resampled['y'].shift(2)
            df_resampled['lag_3'] = df_resampled['y'].shift(3)
            df_resampled = df_resampled.dropna().reset_index(drop=True)

            total_periods = len(df_resampled)
            if total_periods < 5:
                st.warning("Not enough periods after lag creation. Provide more data.")
                st.stop()

            train_size = total_periods - forecast_steps
            if train_size <= 0:
                st.warning("Not enough data for the selected Forecast Steps.")
                st.stop()

            df_train = df_resampled.iloc[:train_size].copy()
            df_test = df_resampled.iloc[train_size:].copy()

            features = ['day_of_week', 'is_weekend', 'month', 'is_holiday', 'lag_1', 'lag_2', 'lag_3']
            X_train, y_train = df_train[features], df_train['y']
            X_test, y_test = df_test[features], df_test['y']

            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred_test = model.predict(X_test)
            df_last_week = pd.DataFrame({
                'ds': df_test['ds'],
                'Actual': y_test.values,
                'Predicted': y_pred_test
            }).set_index('ds')
            # Compute 95% CI around XGBoost predictions using residual std dev
            residuals = df_last_week['Actual'] - df_last_week['Predicted']
            residual_std = residuals.rolling(window=3, min_periods=1).std().fillna(0)
            df_last_week['ci_lower'] = df_last_week['Predicted'] - 1.96 * residual_std
            df_last_week['ci_upper'] = df_last_week['Predicted'] + 1.96 * residual_std

            # --- Metrics ---
            mae = mean_absolute_error(df_last_week['Actual'], df_last_week['Predicted'])
            rmse = mean_squared_error(df_last_week['Actual'], df_last_week['Predicted']) ** 0.5
            mape = np.mean(
                np.abs((df_last_week['Actual'] - df_last_week['Predicted']) /
                       df_last_week['Actual'].replace(0, np.nan))
            ) * 100 if not df_last_week['Actual'].eq(0).all() else 0
            rolling_change = np.mean(np.abs(df_resampled['y'].diff(1)))

            st.subheader("📈 XGBoost Accuracy Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{mae:.2f}")
            c2.metric("RMSE", f"{rmse:.2f}")
            c3.metric("MAPE", f"{mape:.2f}%")
            c4.metric("Rolling Change", f"{rolling_change:.2f}")

            forecast_summary = {
    "machine": selected_machine,
    "trend": "increasing" if rolling_change > 0 else "decreasing" if rolling_change < 0 else "flat",
    "weekend_peak": df_resampled["is_weekend"].tail(7).mean() > 0.5,
    "holiday_next_week": df_resampled["is_holiday"].tail(7).mean() > 0.3,
    "last_week_avg_sales": df_resampled["y"].iloc[-7:].mean()
}

            forecast_input_for_future = df_resampled.copy()
            last_known_date = forecast_input_for_future['ds'].iloc[-1]
            future_dates = pd.date_range(
                start=last_known_date + pd.tseries.frequencies.to_offset(selected_freq),
                periods=forecast_steps,
                freq=selected_freq
            )

            forecast_list = []
            for date in future_dates:
                feat_dict = {
                    'day_of_week': date.dayofweek,
                    'is_weekend': int(date.dayofweek in [5, 6]),
                    'month': date.month,
                    'is_holiday': int(date in public_holidays),
                    'lag_1': forecast_input_for_future['y'].iloc[-1],
                    'lag_2': forecast_input_for_future['y'].iloc[-2] if len(forecast_input_for_future) >= 2 else forecast_input_for_future['y'].mean(),
                    'lag_3': forecast_input_for_future['y'].iloc[-3] if len(forecast_input_for_future) >= 3 else forecast_input_for_future['y'].mean()
                }
                X_future = pd.DataFrame([feat_dict])
                y_future = model.predict(X_future)[0]
                forecast_list.append({'ds': date, 'Forecast': y_future})
                forecast_input_for_future = pd.concat(
                    [forecast_input_for_future, pd.DataFrame({'ds': [date], 'y': [y_future]})],
                    ignore_index=True
                )

            df_pred = pd.DataFrame(forecast_list).set_index('ds')

            # Slider for display history
            unit_label = {"D": "day", "W": "week", "M": "month"}[selected_freq]
            max_history = len(df_resampled)
            default_show = min(30, max_history)
            history_window = st.slider(
                f"🗂️ Display last N {unit_label}s of history",
                min_value=2,
                max_value=max_history,
                value=default_show
            )

            cutoff_date = df_resampled['ds'].iloc[-history_window]
            df_train_plot = df_train[df_train['ds'] >= cutoff_date]
            df_test_plot = df_test[df_test['ds'] >= cutoff_date]
            df_last_week_plot = df_last_week[df_last_week.index >= cutoff_date]

            fig = plot_xgb_forecast_chart(
                df_train_plot, df_test_plot, df_last_week_plot, df_pred,
                selected_machine, freq
            )
            st.pyplot(fig)

            if st.checkbox("🧠 Show AI Recommendations Based on Forecast"):
                with st.spinner("🧠 Thinking... generating suggestions..."):
                    try:
                        suggestions = generate_llm_recommendation(forecast_summary)
                        st.subheader("💡 AI-Generated Operational Suggestions")
                        st.markdown(suggestions)
                    except Exception as e:
                        st.error(f"❌ LLM error: {e}")

            st.download_button(
                label="📥 Download Forecast CSV",
                data=df_pred.reset_index().to_csv(index=False).encode(),
                file_name=f"{selected_machine}_{freq.lower()}_xgboost_forecast.csv",
                mime="text/csv"
            )
        else:
            st.error("Required columns 'locationId' and 'Qty' not found in the uploaded file.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.warning("⚠️ Please upload a file from the Home tab to begin.")
