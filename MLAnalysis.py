import streamlit as st 
import pandas as pd 
import os
#Importing Profiling 
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
#Importing ML needs
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-03.png")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Analytics", "ML-training", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit")
 
st.write("Hey There, I hope you're doing great")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
   st.title("Upload your Data for Analysis please")
   file = st.file_uploader("Upload Your Dataset Right Here")
   if file:
       df = pd.read_csv(file, index_col=None)
       df.to_csv("sourcedata.csv", index=None)
       st.dataframe(df)
       
if choice == "Analytics": 
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()  
    st_profile_report(profile_report)

if choice == "ML-training":
    st.title("ML Model Doing something cool... I guess")
    target = st.selectbox("select your target", df.columns)
    if st.button("Train model"):
     setup(df, target=target, verbose=False)
     setup_df = pull()
     st.info("This is the ML experiment settings")
     st.dataframe(setup_df)
     best_model = compare_models()
     compare_df = pull()
     st.info("This is ML model")
     st.dataframe(compare_df)
     best_model 
     save_model(best_model, "best_model")

if choice == "Download":
    with open("best_model.pkl", "rb") as f: 
        st.download_button("Download the model", f, "train_model_pkl")
