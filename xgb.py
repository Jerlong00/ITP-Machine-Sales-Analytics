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

        df['day_of_week'] = df['saleDate'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['saleDate'].dt.month

        public_holidays = pd.to_datetime([
            "2023-01-01", "2023-01-02", "2023-01-22", "2023-01-23", "2023-01-24",
            "2023-04-07", "2023-04-22", "2023-05-01", "2023-06-02", "2023-06-29",
            "2023-08-09", "2023-09-01", "2023-11-12", "2023-11-13", "2023-12-25",
            "2024-01-01", "2024-02-10", "2024-02-11", "2024-03-29", "2024-04-10",
            "2024-05-01", "2024-05-22", "2024-06-17", "2024-08-09", "2024-10-31", "2024-12-25",
            "2025-01-01", "2025-01-29", "2025-01-30", "2025-03-31", "2025-04-18",
            "2025-05-01", "2025-05-03", "2025-05-12"
        ])
        df['is_holiday'] = df['saleDate'].isin(public_holidays).astype(int)

        if 'locationId' in df.columns and 'Qty' in df.columns:
            machines = df['locationId'].unique().tolist()
            selected_machine = st.selectbox("🛠 Select machine to forecast:", machines)
            model_type = st.radio("📊 Select model:", ["SARIMAX", "XGBoost"])

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
            df_machine = df[df['locationId'] == selected_machine].copy()
            df_machine.set_index('saleDate', inplace=True)
            df_resampled = df_machine['Qty'].resample(selected_freq).sum().to_frame()
            df_resampled.reset_index(inplace=True)
            df_resampled.rename(columns={'saleDate': 'ds', 'Qty': 'y'}, inplace=True)

            df_resampled['day_of_week'] = df_resampled['ds'].dt.dayofweek
            df_resampled['is_weekend'] = df_resampled['day_of_week'].isin([5, 6]).astype(int)
            df_resampled['month'] = df_resampled['ds'].dt.month
            df_resampled['is_holiday'] = df_resampled['ds'].isin(public_holidays).astype(int)

            forecast_steps = st.slider("⏱️ Forecast Steps (Periods)", 4, 100, 12)
            use_boxcox = st.checkbox("📐 Apply Box-Cox transformation (stabilize variance)", value=False)

            if model_type == "SARIMAX":
                seasonal_periods = {"D": 7, "W": 52, "M": 12, "Q": 4, "2Q": 2, "Y": 1}
                seasonal_m = seasonal_periods[selected_freq]

                z_scores = zscore(df_resampled['y'].bfill())
                threshold = 3
                df_resampled['cleaned'] = df_resampled['y']
                df_resampled.loc[(z_scores > threshold) | (z_scores < -threshold), 'cleaned'] = np.nan
                df_resampled['cleaned'].interpolate(method="linear", inplace=True)
                df_resampled['cleaned_filled'] = df_resampled['cleaned'].bfill().clip(lower=0.01)

                if use_boxcox:
                    df_resampled['transformed'], lam = boxcox(df_resampled['cleaned_filled'])
                    series_to_use = df_resampled['transformed']
                else:
                    series_to_use = df_resampled['cleaned_filled']

                train = series_to_use.iloc[:-forecast_steps]
                test = df_resampled['y'].iloc[-forecast_steps:]

                try:
                    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,seasonal_m))
                    results = model.fit(disp=False)
                    predicted = results.forecast(steps=forecast_steps)
                    if use_boxcox:
                        predicted = inv_boxcox(predicted, lam)

                    mae = mean_absolute_error(test, predicted)
                    rmse = mean_squared_error(test, predicted) ** 0.5
                    mape = np.mean(np.abs((test - predicted) / test.replace(0, np.nan))) * 100

                    st.subheader("📊 SARIMAX Accuracy Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MAE", f"{mae:.2f}")
                    col2.metric("RMSE", f"{rmse:.2f}")
                    col3.metric("MAPE", f"{mape:.2f}%")

                    st.subheader("📈 SARIMAX Forecast")
                    fig, ax = plt.subplots()
                    df_resampled['y'].plot(ax=ax, label='Historical')
                    pd.Series(predicted, index=test.index).plot(ax=ax, label='Forecast', linestyle="--")
                    ax.legend()
                    st.pyplot(fig)

                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=pd.Series(predicted, index=test.index).to_csv().encode(),
                        file_name=f"{selected_machine}_{freq.lower()}_sarimax_forecast.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"SARIMAX failed: {e}")

            else:
                df_daily = df_machine['Qty'].resample('D').sum().reset_index()
                df_daily.rename(columns={'saleDate': 'ds', 'Qty': 'y'}, inplace=True)

                df_daily['day_of_week'] = df_daily['ds'].dt.dayofweek
                df_daily['is_weekend'] = df_daily['day_of_week'].isin([5, 6]).astype(int)
                df_daily['month'] = df_daily['ds'].dt.month
                df_daily['is_holiday'] = df_daily['ds'].isin(public_holidays).astype(int)

                df_daily['lag_1'] = df_daily['y'].shift(1)
                df_daily['lag_7'] = df_daily['y'].shift(7)
                df_daily['lag_14'] = df_daily['y'].shift(14)
                df_daily.dropna(inplace=True)

                features = ['day_of_week', 'is_weekend', 'month', 'is_holiday', 'lag_1', 'lag_7', 'lag_14']
                X = df_daily[features]
                y = df_daily['y']
                dates = df_daily['ds']

                X_train, y_train = X.iloc[:-forecast_steps], y.iloc[:-forecast_steps]
                X_test, y_test = X.iloc[-forecast_steps:], y.iloc[-forecast_steps:]
                dates_test = dates.iloc[-forecast_steps:]

                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                df_pred = pd.DataFrame({'ds': dates_test, 'Forecast': preds})
                df_pred.set_index('ds', inplace=True)
                df_pred = df_pred.resample(selected_freq).sum()

                df_true = pd.DataFrame({'ds': dates_test, 'y': y_test.values})
                df_true.set_index('ds', inplace=True)
                df_true = df_true.resample(selected_freq).sum()

                mae = mean_absolute_error(df_true['y'], df_pred['Forecast'])
                rmse = mean_squared_error(df_true['y'], df_pred['Forecast']) ** 0.5
                mape = np.mean(np.abs((df_true['y'] - df_pred['Forecast']) / df_true['y'].replace(0, np.nan))) * 100

                st.subheader("📊 XGBoost Accuracy Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{mae:.2f}")
                col2.metric("RMSE", f"{rmse:.2f}")
                col3.metric("MAPE", f"{mape:.2f}%")

                st.subheader("📈 XGBoost Forecast")
                fig, ax = plt.subplots(figsize=(10, 4))
                df_true['y'].plot(ax=ax, label='Actual', color='blue', marker='o')
                df_pred['Forecast'].plot(ax=ax, label='Forecast', linestyle='--', color='orange', marker='x')
                ax.set_ylim(min(df_true['y'].min(), df_pred['Forecast'].min()) * 0.95,
                            max(df_true['y'].max(), df_pred['Forecast'].max()) * 1.05)
                ax.legend()
                st.pyplot(fig)

                st.download_button(
                    label="📥 Download XGBoost Forecast CSV",
                    data=df_pred.to_csv().encode(),
                    file_name=f"{selected_machine}_{freq.lower()}_xgboost_forecast.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
