import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Sales Forecasting", layout="wide")
st.title("📤 Upload Your Sales File")


# Upload only on mainapp and share with all models
uploaded_file = st.file_uploader("Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Clean up column names
        df.columns = df.columns.astype(str).str.strip().str.replace('\xa0','').str.replace(':','')

        # ✅ Store in session for use in all models
        st.session_state["df"] = df

        st.success("File uploaded and saved in session. All models can now access it from their tabs.")
        st.write(df.head())
    except Exception as e:
        st.error(f"❌ Something went wrong while reading your file: {e}")
else:
    if "df" in st.session_state:
        st.info("✅ File already uploaded. You can switch tabs to proceed.")
        st.write(st.session_state["df"].head())
    else:
        st.warning("⚠️ Please upload a file to begin forecasting.")
