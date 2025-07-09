import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

st.set_page_config(page_title="AI Sales Forecasting", layout="wide")
st.title("📈 AI Sales Forecasting App (Multi-Frequency)")

st.markdown("✅ Upload your Excel file with daily machine sales data. Public holiday support coming soon!")

uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # Load Excel
        xls = pd.ExcelFile(uploaded_file)
        df = pd.read_excel(xls, sheet_name=0)

        # 🔧 CLEAN COLUMN NAMES
        df.columns = df.columns.astype(str)               # ensure all column names are strings
        df.columns = df.columns.str.strip()               # strip whitespace
        df.columns = df.columns.str.replace('\xa0', '')   # remove non-breaking spaces
        df.columns = df.columns.str.replace(':', r'\:')   # escape colons for Altair

        # 🕵️ Show the columns for debugging
        st.write("📋 Columns found in uploaded file:")
        st.write(df.columns.tolist())

        # 🔍 Detect date column
        date_column_candidates = [col for col in df.columns if 'date' in col.lower()]
        if not date_column_candidates:
            st.warning("⚠️ Couldn’t find a 'date' column — defaulting to first column")
            date_column_candidates = [df.columns[0]]

        selected_date_col = st.selectbox("📅 Select the date column:", date_column_candidates)
        df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
        df = df.dropna(subset=[selected_date_col])
        df.set_index(selected_date_col, inplace=True)

        # 🛠️ Detect machine columns
        machine_cols = [col for col in df.columns if col.upper().startswith("A")]
        if not machine_cols:
            st.error("❌ No machine columns found. Expected columns like A1, A2, A3...")
            st.stop()

        selected_machine = st.selectbox("🛠️ Select machine to forecast:", machine_cols)

        # ⏱️ Frequency selection
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
        df_resampled = df[[selected_machine]].resample(selected_freq).sum()

        # ✅ Plot with Matplotlib (no Altair errors)
        st.subheader(f"📉 {selected_machine} Sales Over Time ({freq})")
        fig, ax = plt.subplots()
        df_resampled[selected_machine].plot(ax=ax, label="Historical")
        ax.set_ylabel("Sales")
        ax.legend()
        st.pyplot(fig)

        # ⏩ Forecast period
        step_defaults = {
            "Daily": 30,
            "Weekly": 12,
            "Monthly": 12,
            "Quarterly": 8,
            "Bi-annually": 4,
            "Yearly": 3
        }

        forecast_steps = st.slider(
            f"🔮 Forecast how many {freq.lower()} periods?",
            2, step_defaults[freq]*2, step_defaults[freq]
        )

        # 🧠 SARIMAX
        model = SARIMAX(df_resampled[selected_machine], order=(1,1,1), seasonal_order=(1,1,1,4))
        results = model.fit(disp=False)

        forecast = results.get_forecast(steps=forecast_steps)
        forecast_df = forecast.predicted_mean.rename(f"{selected_machine}_forecast")

        # 📈 Forecast Plot
        st.subheader("📊 Forecast vs Historical")
        fig2, ax2 = plt.subplots()
        df_resampled[selected_machine].plot(ax=ax2, label="Historical")
        forecast_df.plot(ax=ax2, label="Forecast", linestyle="--")
        ax2.set_ylabel("Sales")
        ax2.legend()
        st.pyplot(fig2)

        # 💾 Download
        st.download_button(
            label="📥 Download Forecast CSV",
            data=forecast_df.to_csv().encode(),
            file_name=f"{selected_machine}_{freq.lower()}_forecast.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
