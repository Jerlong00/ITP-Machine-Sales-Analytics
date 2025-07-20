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

st.set_page_config(page_title="AI Sales Forecasting", layout="wide")
st.title("📈 AI Sales Forecasting App (Multi-Frequency)")

st.markdown("✅ Upload your Excel or CSV file with machine sales data. Public holiday support coming soon!")

uploaded_file = st.file_uploader("📤 Upload Excel or CSV File", type=["xlsx", "xls", "csv"], key="xgb_uploader")

if uploaded_file:
    try:
        filename = uploaded_file.name.lower()
        df = pd.read_csv(uploaded_file) if filename.endswith(".csv") else pd.read_excel(uploaded_file)
        df.columns = df.columns.astype(str).str.strip().str.replace('\xa0', '').str.replace(':', ':')

        df['saleDate'] = pd.to_datetime(df['saleDate'])
        df = df.dropna(subset=['saleDate'])

        public_holidays = pd.to_datetime([
            "2023-01-01", "2023-01-02", "2023-01-22", "2023-01-23", "2023-01-24",
            "2023-04-07", "2023-04-22", "2023-05-01", "2023-06-02", "2023-06-29",
            "2023-08-09", "2023-09-01", "2023-11-12", "2023-11-13", "2023-12-25",
            "2024-01-01", "2024-02-10", "2024-02-11", "2024-03-29", "2024-04-10",
            "2024-05-01", "2024-05-22", "2024-06-17", "2024-08-09", "2024-10-31", "2024-12-25",
            "2025-01-01", "2025-01-29", "2025-01-30", "2025-03-31", "2025-04-18",
            "2025-05-01", "2025-05-03", "2025-05-12"
        ])

        if 'locationId' in df.columns and 'Qty' in df.columns:
            machines = df['locationId'].unique().tolist()
            selected_machine = st.selectbox("\U0001F6E0 Select machine to forecast:", machines)

            freq = st.selectbox("\U0001F4CA Forecast Frequency:", ["Daily", "Weekly", "Monthly"])
            freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
            selected_freq = freq_map[freq]

            forecast_steps = st.slider("⏱️ Forecast Steps (Periods)", 2, 60, 14)

            df_machine = df[df['locationId'] == selected_machine].copy()
            df_machine.set_index('saleDate', inplace=True)
            df_machine = df_machine.sort_index()

            df_daily = df_machine['Qty'].resample('D').sum().reset_index()
            df_daily.rename(columns={'saleDate': 'ds', 'Qty': 'y'}, inplace=True)

            df_daily['day_of_week'] = df_daily['ds'].dt.dayofweek
            df_daily['is_weekend'] = df_daily['day_of_week'].isin([5, 6]).astype(int)
            df_daily['month'] = df_daily['ds'].dt.month
            df_daily['is_holiday'] = df_daily['ds'].isin(public_holidays).astype(int)

            df_daily['lag_1'] = df_daily['y'].shift(1)
            df_daily['lag_7'] = df_daily['y'].shift(7)
            df_daily['lag_14'] = df_daily['y'].shift(14)
            df_daily = df_daily.dropna()

            features = ['day_of_week', 'is_weekend', 'month', 'is_holiday', 'lag_1', 'lag_7', 'lag_14']
            X = df_daily[features]
            y = df_daily['y']

            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
            model.fit(X, y)

            # Use the last 'forecast_steps' rows to compute last week forecast
            X_test = X.iloc[-forecast_steps:]
            y_test = y.iloc[-forecast_steps:]
            y_pred_test = model.predict(X_test)

            # Append predictions to forecast_input for consistency in future forecasting
            forecast_input = df_daily.copy()
            last_known_date = forecast_input['ds'].iloc[-1]
            for i in range(forecast_steps):
                forecast_input = pd.concat([forecast_input, pd.DataFrame({
                    'ds': [X_test.index[i]],
                    'y': [y_pred_test[i]]
                })], ignore_index=True)

            # Forecast future based on those appended rows
            future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
            forecast_list = []
            for date in future_dates:
                features_dict = {
                    'day_of_week': date.dayofweek,
                    'is_weekend': int(date.dayofweek in [5, 6]),
                    'month': date.month,
                    'is_holiday': int(date in public_holidays),
                    'lag_1': forecast_input['y'].iloc[-1],
                    'lag_7': forecast_input['y'].iloc[-7] if len(forecast_input) >= 7 else forecast_input['y'].mean(),
                    'lag_14': forecast_input['y'].iloc[-14] if len(forecast_input) >= 14 else forecast_input['y'].mean()
                }
                X_pred = pd.DataFrame([features_dict])
                y_pred = model.predict(X_pred)[0]
                forecast_list.append({'ds': date, 'Forecast': y_pred})
                forecast_input = pd.concat([forecast_input, pd.DataFrame({'ds': [date], 'y': [y_pred]})], ignore_index=True)

            df_pred = pd.DataFrame(forecast_list).set_index('ds')
            df_pred.index = pd.to_datetime(df_pred.index)
            df_pred = df_pred.resample(selected_freq).sum()

            df_recent = df_daily.set_index('ds')
            df_recent = df_recent[df_recent.index >= df_recent.index.max() - pd.Timedelta(days=30)]
            df_recent_resampled = df_recent.resample(selected_freq).sum()

            df_last_week = pd.DataFrame({
                'ds': df_daily['ds'].iloc[-forecast_steps:],
                'Actual': y_test.values,
                'Predicted': y_pred_test
            })
            df_last_week.set_index('ds', inplace=True)
            df_last_week.index = pd.to_datetime(df_last_week.index)
            df_last_week_resampled = df_last_week.resample(selected_freq).sum()

            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
            mape = np.mean(np.abs((y_test - y_pred_test) / y_test.replace(0, np.nan))) * 100
            rolling_mae = np.mean(np.abs(df_daily['y'].diff(periods=7)))

            st.subheader("\U0001F4CA XGBoost Accuracy Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("MAPE", f"{mape:.2f}%")
            col4.metric("Rolling MAE (7d)", f"{rolling_mae:.2f}")

            st.subheader("\U0001F4C8 XGBoost Forecast")
            fig, ax = plt.subplots(figsize=(10, 4))
            df_recent_resampled['y'].plot(ax=ax, label='Past 1 Month', marker='o', color='blue')
            df_last_week_resampled['Predicted'].plot(ax=ax, label='Forecast (last week)', linestyle='--', marker='x', color='orange')
            df_pred['Forecast'].plot(ax=ax, label='Forecast (future)', linestyle='--', marker='x', color='green')
            ax.legend()
            st.pyplot(fig)

            st.download_button(
                label="\U0001F4C5 Download XGBoost Forecast CSV",
                data=df_pred.to_csv().encode(),
                file_name=f"{selected_machine}_{freq.lower()}_xgboost_forecast.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
