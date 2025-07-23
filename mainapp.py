import streamlit as st

# App setup
st.set_page_config(page_title="AI Sales Forecasting", layout="wide")
st.title("📈 AI Sales Forecasting App")

# Tab navigation for different forecasting models
tabs = st.tabs(["Home", "SARIMAX", "Prophet", "XGBoost", "Rolling Naive"])

with tabs[0]:
    st.subheader("Welcome to the AI Sales Forecasting App 🎯")
    st.markdown("""
        Use the tabs above to navigate between different forecasting models:

        - **SARIMAX**: Seasonal ARIMA with exogenous regressors.
        - **Prophet**: Facebook Prophet model with holidays and school breaks.
        - **XGBoost**: Machine learning-based time series model.
        - **Rolling Naive**: Baseline forecast using past observations.

        Each model page will guide you through uploading your data, selecting frequency,
        choosing a machine, and generating forecasts with accuracy metrics and plots.

        ---
        Please proceed to one of the model tabs to begin.
    """)


