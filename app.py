import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model, load_model, create_app

with st.sidebar:
    st.image(
        "https://www.onepointltd.com/wp-content/uploads/2019/12/ONE-POINT-01-1.png"
        # "https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png"
    )
    st.title("AutoML")
    st.caption("This a basic app for automated model training")
    choice = st.radio("Navigation", ["Data Upload", "Data Profiling", "Model Training", "Evaluate"])

if choice == "Data Upload":
    st.title("Data Upload")
    st.text("Upload tabular data for which you would like to train a model")
    file = st.file_uploader("Upload your dataset here !")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("./data/dataset.csv", index=None)
        st.dataframe(df.head(20))

if choice == "Data Profiling":
    st.title("Data Profiling")
    st.text("Automated Exploratory Data Analysis")

    if not os.path.exists("./data/dataset.csv"):
        st.error("Please upload a dataset to be profiled !")

    else:
        df_p = pd.read_csv("./data/dataset.csv", index_col=None)
        profile_report = df_p.profile_report()
        st_profile_report(profile_report)


if choice == "Model Training":
    st.title("Model Training")
    df_ml = pd.read_csv("./data/dataset.csv", index_col=None)
    target = st.selectbox("Select target variable", df_ml.columns)
    if st.button("Train Model"):
        setup(df_ml, target=target, normalize=True, transformation=True, log_experiment=True)
        setup_df = pull()
        st.info("These are the ML experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.divider()
        st.info("Model Leaderboard")
        st.dataframe(compare_df)
        save_model(best_model, "./best_model/best_model")
        with open("./best_model/best_model.pkl", "rb") as f:
            st.download_button("Download Model", f, "trained_model.pkl")

if choice == "Evaluate":
    st.title("Evaluate")
    st.text("Evaluate the best performing model")
    pipeline = load_model("./best_model/best_model")
    # df_eval = pd.read_csv("./data/dataset.csv", index_col=None)
    # setup(data=df_eval, target="Survived")
    # create_app(pipeline)