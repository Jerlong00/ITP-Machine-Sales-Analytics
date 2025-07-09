import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

st.title("📈 AI Sales Forecasting App")

# File uploader
uploaded_file = st.file_uploader("Upload your sales data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    if 'date' in df.columns and 'sales' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        forecast_days = st.slider("Forecast how many days ahead?", 7, 60, 30)

        model = SARIMAX(df['sales'], order=(1,1,1), seasonal_order=(1,1,1,12))
        results = model.fit(disp=False)

        forecast = results.get_forecast(steps=forecast_days)
        forecast_df = forecast.predicted_mean

        # Plot
        st.subheader("Forecast vs Historical Sales")
        fig, ax = plt.subplots()
        df['sales'].plot(ax=ax, label='Historical')
        forecast_df.plot(ax=ax, label='Forecast')
        ax.legend()
        st.pyplot(fig)

        # Downloadable forecast
        st.download_button(
            "Download Forecast",
            data=forecast_df.to_csv().encode(),
            file_name="forecast.csv",
            mime="text/csv"
        )
    else:
        st.error("CSV must have 'date' and 'sales' columns!")
