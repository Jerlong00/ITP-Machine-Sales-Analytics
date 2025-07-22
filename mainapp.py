import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import zscore, boxcox
from scipy.special import inv_boxcox
import numpy as np
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Sales Forecasting", layout="wide")
st.title("📈 AI Sales Forecasting App (Multi-Frequency)")
st.markdown("✅ Upload your Excel or CSV file with machine sales data.")

uploaded_file = st.file_uploader("📤 Upload Excel or CSV File", type=["xlsx","xls","csv"])

if uploaded_file:
    try:
        # --- Load & clean ---
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, sheet_name=0)
        df.columns = df.columns.astype(str).str.strip().str.replace('\xa0','').str.replace(':',r'\:')
        
        # --- Date index selection ---
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if not date_cols:
            st.warning("⚠️ No date column found, defaulting to first column.")
            date_cols = [df.columns[0]]
        sel_date = st.selectbox("📅 Select the date column:", date_cols)
        df[sel_date] = pd.to_datetime(df[sel_date], errors="coerce")
        df = df.dropna(subset=[sel_date]).set_index(sel_date)
        
        # filter out flagged outliers if present
        if "outlier_flag" in df.columns:
            df = df[df["outlier_flag"]==0]
        
        # --- Frequency & pivot ---
        freq = st.selectbox("📊 Forecast Frequency:",
                            ["Daily","Weekly","Monthly","Quarterly","Bi-annually","Yearly"])
        freq_map = {"Daily":"D","Weekly":"W","Monthly":"M","Quarterly":"Q","Bi-annually":"2Q","Yearly":"Y"}
        sel_freq = freq_map[freq]
        
        if "locationId" in df.columns and "Qty" in df.columns:
            df = df[df["Qty"]>0]
            piv = df.pivot_table(index=df.index, columns="locationId", values="Qty", aggfunc="sum")
            machines = piv.columns.astype(str).tolist()
            sel_mach = st.selectbox("🛠 Select machine to forecast:", machines)
            resamp = "sum" if freq in ["Daily","Weekly"] else "mean"
            df_res = piv[sel_mach].resample(sel_freq).agg(resamp).to_frame(name=sel_mach)
        else:
            st.error("❌ Your file needs 'locationId' & 'Qty' columns.")
            st.stop()
        
        # --- Outlier smoothing & Box-Cox ---
        use_bc = st.checkbox("📐 Apply Box-Cox", value=False)
        zs = zscore(df_res[sel_mach].fillna(method="bfill"))
        df_res["clean"] = df_res[sel_mach]
        df_res.loc[(zs>3)|(zs<-3),"clean"] = np.nan
        df_res["clean"].interpolate(inplace=True)
        df_res["clean_filled"] = df_res["clean"].bfill().clip(lower=0.01)
        if use_bc:
            df_res["trans"], lam = boxcox(df_res["clean_filled"])
            series_to_use = df_res["trans"]
        else:
            series_to_use = df_res["clean_filled"]
        
        # --- Forecast horizon ---
        defaults = {"Daily":30,"Weekly":12,"Monthly":12,"Quarterly":8,"Bi-annually":4,"Yearly":3}
        h = st.slider(f"📆 How many {freq.lower()} periods to forecast?",
                      2, defaults[freq]*2, defaults[freq])
        
        # --- Model toggle ---
        model_choice = st.radio("🧠 Select Forecasting Model:", ["SARIMAX","Prophet"])
        
        # Prophet branch → delegate & stop
        if model_choice == "Prophet":
            from prophet_forecast import run_prophet_forecast

            df_res_prophet = df_res[[sel_mach]].rename(columns={sel_mach: "y"})
            df_res_prophet["ds"] = df_res_prophet.index

            run_prophet_forecast(
                df_resampled=df_res,
                series_to_use=series_to_use,
                forecast_steps=h,
                selected_freq=sel_freq,
                selected_machine=sel_mach,
                freq=freq
            )
            st.stop()
        
        # ————— SARIMAX branch —————
        seasonal_m = {"D":7,"W":52,"M":12,"Q":4,"2Q":2,"Y":1}[sel_freq]
        train = series_to_use.iloc[:-h]
        test  = df_res[sel_mach].iloc[-h:]
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,seasonal_m))
        res   = model.fit(disp=False)
        preds = res.forecast(steps=h)
        if use_bc:
            preds = inv_boxcox(preds, lam)
        preds = pd.Series(preds, index=test.index)
        
        # ————— Accuracy metrics —————
        actual = test
        rmse   = mean_squared_error(actual,preds)**0.5
        mae    = mean_absolute_error(actual,preds)
        mape   = np.mean(np.abs((actual-preds)/actual.replace(0,np.nan)))*100
        rolling= np.mean(np.abs(df_res[sel_mach].diff()))
        
        st.subheader("📊 Forecast Accuracy Metrics")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("MAE",f"{mae:.2f}")
        c2.metric("RMSE",f"{rmse:.2f}")
        c3.metric("MAPE",f"{mape:.2f}%")
        c4.metric("Rolling MAE",f"{rolling:.2f}")
        
        # ————— Plot —————
        st.subheader("📈 Forecast vs Historical")
        fig,ax = plt.subplots()
        df_res[sel_mach].plot(ax=ax,label="Historical")
        preds.plot(ax=ax, label="SARIMAX Forecast", linestyle="--")
        ax.set_ylabel("Sales")
        ax.legend()
        st.pyplot(fig)
        
        # Download
        st.download_button(
            "📥 Download SARIMAX Forecast",
            data=preds.to_csv().encode(),
            file_name=f"{sel_mach}_{freq.lower()}_sarimax_forecast.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")


