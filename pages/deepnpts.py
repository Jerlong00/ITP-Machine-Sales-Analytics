# deepnpts_forecast_app.py
# ----------------------------------------------------------
# Streamlit app for multi-frequency sales forecasting
# using Nixtla’s DeepNPTS model
# — now with the same calendar exogenous features
#   as the XGBoost version and a proper back-test window.
# ----------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from neuralforecast import NeuralForecast
from neuralforecast.models import DeepNPTS

def deepnpts_chart_style(ax, title):
    ax.set_title(title, fontsize=18)
    ax.set_ylabel("Qty", fontsize=14)
    ax.set_xlabel("Date", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(True)

st.set_page_config(page_title="AI Sales Forecasting (DeepNPTS)", layout="wide")
st.title("🤖📈 AI Sales Forecasting App – DeepNPTS Edition")
st.markdown(
    "⚙️ Powered by **DeepNPTS** (NeuralForecast), using exogenous features and back-testing."
)

# ---------- Static holiday list ----------
PUBLIC_HOLIDAYS = pd.to_datetime([
    "2023-01-01","2023-01-22","2023-04-07","2023-05-01","2023-08-09",
    "2024-01-01","2024-02-10","2024-03-29","2024-05-01","2024-08-09",
    "2025-01-01","2025-01-29","2025-04-18","2025-05-01",
])

# ---------- Helper for input_size ----------
def suggest_input_size(freq_code: str) -> int:
    return {"D":14, "W":8, "M":6}.get(freq_code, 14)

# ---------- Use shared file ----------
if "df" in st.session_state:
    df = st.session_state["df"]
    try:
        df.columns = df.columns.astype(str).str.strip().str.replace("\xa0","", regex=False)

        must_have = {"saleDate","locationId","Qty"}
        if not must_have.issubset(df.columns):
            st.error("File must contain columns: saleDate, locationId, Qty")
            st.stop()

        df["saleDate"] = pd.to_datetime(df["saleDate"], errors="coerce")
        df = df.dropna(subset=["saleDate"]).sort_values("saleDate")

        machines = sorted(df["locationId"].unique())
        selected_machine = st.selectbox("🔧 Select machine:", machines)

        freq_label = st.selectbox("📊 Forecast Frequency:", ["Daily","Weekly","Monthly"])
        freq_code = {"Daily":"D","Weekly":"W","Monthly":"M"}[freq_label]

        h = st.slider("⏱️ Forecast Steps (back-test + future)", 2, 60, 14)

        df_all = df.set_index("saleDate")[["locationId","Qty"]]
        panel = (
            df_all
            .groupby("locationId")
            .resample(freq_code)["Qty"]
            .sum()
            .reset_index()
            .rename(columns={"locationId":"unique_id","saleDate":"ds","Qty":"y"})
        )

        panel["day_of_week"] = panel["ds"].dt.dayofweek
        panel["is_weekend"] = panel["day_of_week"].isin([5,6]).astype(int)
        panel["month"]       = panel["ds"].dt.month
        panel["is_holiday"]  = panel["ds"].isin(PUBLIC_HOLIDAYS).astype(int)

        counts = panel.groupby("unique_id").size()
        if counts[selected_machine] < h+1:
            st.error(
                f"Machine {selected_machine} only has {counts[selected_machine]} "
                f"{freq_label.lower()} points but you asked for {h}. "
                "Reduce the slider or pick another machine."
            )
            st.stop()

        df_test  = panel.groupby("unique_id").tail(h)
        df_train = pd.concat([panel, df_test]).drop_duplicates(keep=False)

        future_rows = []
        offset = pd.tseries.frequencies.to_offset(freq_code)
        for uid in machines:
            last = panel[panel["unique_id"]==uid]["ds"].max()
            future_dates = pd.date_range(start=last+offset, periods=h, freq=freq_code)
            for d in future_dates:
                future_rows.append({
                    "unique_id": uid,
                    "ds":         d,
                    "day_of_week": d.dayofweek,
                    "is_weekend":  int(d.dayofweek in [5,6]),
                    "month":       d.month,
                    "is_holiday":  int(d in PUBLIC_HOLIDAYS),
                })
        df_future_exog = pd.DataFrame(future_rows)

        exog_vars = ["day_of_week","is_weekend","month","is_holiday"]
        model = DeepNPTS(
            h=h,
            input_size=suggest_input_size(freq_code),
            hist_exog_list=exog_vars,
            futr_exog_list=exog_vars,
            max_steps=300
        )

        nf = NeuralForecast(models=[model], freq=freq_code)
        nf.fit(df_train)
        test_forecasts = nf.predict(futr_df=df_test)

        test_sel = df_test[df_test["unique_id"]==selected_machine].set_index("ds")
        pred_sel = (
            test_forecasts[test_forecasts["unique_id"]==selected_machine]
            .set_index("ds")
            .rename(columns={"DeepNPTS":"Predicted"})
        )
        df_eval = test_sel[["y"]].join(pred_sel, how="inner").rename(columns={"y":"Actual"})

        if df_eval.empty:
            st.warning("⚠️ Not enough overlap to compute back-test metrics.")
        else:
            mae  = mean_absolute_error(df_eval["Actual"], df_eval["Predicted"])
            rmse = np.sqrt(mean_squared_error(df_eval["Actual"], df_eval["Predicted"]))
            mape = np.mean(
                np.abs((df_eval["Actual"]-df_eval["Predicted"])
                       / df_eval["Actual"].replace(0, np.nan))
            )*100
            mape = 0.0 if np.isnan(mape) else mape

            rolling = np.mean(
                np.abs(panel[panel["unique_id"]==selected_machine]["y"].diff())
            )

            st.subheader("📈 DeepNPTS Accuracy Metrics")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("MAE",  f"{mae:.2f}")
            c2.metric("RMSE", f"{rmse:.2f}")
            c3.metric("MAPE", f"{mape:.2f}%")
            c4.metric("Rolling Change", f"{rolling:.2f}")

        model_full = DeepNPTS(
            h=h,
            input_size=suggest_input_size(freq_code),
            hist_exog_list=exog_vars,
            futr_exog_list=exog_vars,
            max_steps=300
        )
        nf_full = NeuralForecast(models=[model_full], freq=freq_code)
        nf_full.fit(panel)
        future_preds = nf_full.predict(futr_df=df_future_exog)

        fut_sel = (
            future_preds[future_preds["unique_id"]==selected_machine]
            .set_index("ds")
            .rename(columns={"DeepNPTS":"Forecast"})
        )

        hist_all = panel[panel["unique_id"]==selected_machine]
        max_hist = len(hist_all)
        unit = {"D":"day", "W":"week", "M":"month"}[freq_code]
        show_n = st.slider(f"🗂️ Display last N {unit}s of history", 2, max_hist, min(30,max_hist))
        cutoff = hist_all["ds"].iloc[-show_n]

        train_vis = df_train[df_train["unique_id"]==selected_machine]
        train_vis = train_vis[train_vis["ds"] >= cutoff]
        test_vis  = df_eval[df_eval.index >= cutoff] if not df_eval.empty else pd.DataFrame()

        st.subheader("📊 DeepNPTS Forecast")
        fig, ax = plt.subplots(figsize=(12,5))

        if not train_vis.empty:
            y_vals = train_vis.set_index("ds")["y"]
            y_vals.plot(ax=ax, label="Training Actuals", color="blue", marker="o")
            for x, y in y_vals.items():
                ax.text(x, y + 0.5, f"{y:.0f}", color="blue", fontsize=9, ha="center")

        if not test_vis.empty:
            actual = test_vis["Actual"]
            pred = test_vis["Predicted"]
            actual.plot(ax=ax, label="Test Actuals", color="purple", marker="o")
            pred.plot(ax=ax, label="Test Predictions", color="orange", linestyle="--", marker="x")
            for x, y in actual.items():
                ax.text(x, y + 0.5, f"{y:.0f}", color="purple", fontsize=9, ha="center")
            for x, y in pred.items():
                ax.text(x, y + 0.5, f"{y:.0f}", color="orange", fontsize=9, ha="center")

        forecast = fut_sel["Forecast"]
        forecast.plot(ax=ax, label="Future Forecast", color="green", linestyle="--", marker="x")
        for x, y in forecast.items():
            ax.text(x, y + 0.5, f"{y:.0f}", color="green", fontsize=9, ha="center")

        deepnpts_chart_style(ax, title=f"{selected_machine} Sales Forecast – {freq_label}")
        st.pyplot(fig)

        st.download_button(
            "📥 Download Forecast CSV",
            data=fut_sel.reset_index().to_csv(index=False).encode(),
            file_name=f"{selected_machine}_{freq_label.lower()}_deepnpts_forecast.csv",
            mime="text/csv"
        )

    except ModuleNotFoundError:
        st.error("NeuralForecast not installed. Run `pip install neuralforecast`.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.warning("⚠️ Please upload a file in the Home tab first.")
