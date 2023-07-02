import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    st.image(
        "https://www.onepointltd.com/wp-content/uploads/2019/12/ONE-POINT-01-1.png"
        # "https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png"
    )
    st.title("AutoML")
    st.caption("This a basic app for automated model training")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])

if choice == "Upload":
    st.title("Upload")
    file = st.file_uploader("Upload your dataset here !")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("./data/dataset.csv", index=None)
        st.dataframe(df.head(20))

if choice == "Profiling":
    st.title("Profiling")
    st.text("Automated Exploratory Data Analysis")

    if not os.path.exists("./data/dataset.csv"):
        st.error("Please upload a dataset to be profiled !")

    df_p = pd.read_csv("./data/dataset.csv", index_col=None)
    profile_report = df_p.profile_report()
    st_profile_report(profile_report)


if choice == "ML":
    st.title("ML")
    df_ml = pd.read_csv("./data/dataset.csv", index_col=None)
    target = st.selectbox("Select target variable", df_ml.columns)
    if st.button("Train Model"):
        setup(df_ml, target=target)
        setup_df = pull()
        st.info("These are the ML experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, "./best_model/best_model")

if choice == "Download":
    st.title("Download")
    with open("./best_model/best_model.pkl", "rb") as f:
        st.download_button("Download Model", f, "trained_model.pkl")
