import streamlit as st
import pandas as pd
import numpy as np
from prophet_forecast import run_prophet_forecast
from scipy.stats import zscore, boxcox

st.set_page_config(page_title="Prophet Forecasting", layout="wide")
st.title("🔮 Prophet Forecasting Module")

uploaded_file = st.file_uploader("📤 Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, sheet_name=0)
        df.columns = df.columns.astype(str).str.strip().str.replace('\xa0', '').str.replace(':', r'\:')

        date_cols = [c for c in df.columns if "date" in c.lower()]
        if not date_cols:
            st.warning("⚠️ No date column found, defaulting to first column.")
            date_cols = [df.columns[0]]
        sel_date = st.selectbox("📅 Select the date column:", date_cols)
        df[sel_date] = pd.to_datetime(df[sel_date], errors="coerce")
        df = df.dropna(subset=[sel_date]).set_index(sel_date)

        if "outlier_flag" in df.columns:
            df = df[df["outlier_flag"] == 0]

        freq = st.selectbox("📊 Forecast Frequency:", ["Daily", "Weekly", "Monthly"])
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        sel_freq = freq_map[freq]

        if "locationId" in df.columns and "Qty" in df.columns:
            df = df[df["Qty"] > 0]
            piv = df.pivot_table(index=df.index, columns="locationId", values="Qty", aggfunc="sum")
            machines = piv.columns.astype(str).tolist()
            sel_mach = st.selectbox("🛠 Select machine to forecast:", machines)
            resamp = "sum" if freq in ["Daily", "Weekly"] else "mean"
            df_res = piv[sel_mach].resample(sel_freq).agg(resamp).to_frame(name=sel_mach)
        else:
            st.error("❌ Your file needs 'locationId' & 'Qty' columns.")
            st.stop()

        use_bc = st.checkbox("📐 Apply Box-Cox", value=False)
        zs = zscore(df_res[sel_mach].fillna(method="bfill"))
        df_res["clean"] = df_res[sel_mach]
        df_res.loc[(zs > 3) | (zs < -3), "clean"] = np.nan
        df_res["clean"].interpolate(inplace=True)
        df_res["clean_filled"] = df_res["clean"].bfill().clip(lower=0.01)
        if use_bc:
            df_res["trans"], lam = boxcox(df_res["clean_filled"])
            series_to_use = df_res["trans"]
        else:
            series_to_use = df_res["clean_filled"]

        defaults = {"Daily": 30, "Weekly": 12, "Monthly": 12, "Quarterly": 8, "Bi-annually": 4, "Yearly": 3}
        h = st.slider(f"📆 How many {freq.lower()} periods to forecast?", 2, defaults[freq]*2, defaults[freq])

        run_prophet_forecast(
            df_resampled=df_res,
            series_to_use=series_to_use,
            forecast_steps=h,
            selected_freq=sel_freq,
            selected_machine=sel_mach,
            freq=freq
        )

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
