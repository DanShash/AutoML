# MLAnalysis.py

import streamlit as st
import pandas as pd
import os

# Profiling and ML
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull as clf_pull, save_model as clf_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save

# Set page config
st.set_page_config(page_title="AutoStreamML", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-03.png", width=250)
    st.title("AutoStreamML")
    st.markdown("📊 AutoML Dashboard for Classification and Regression")
    choice = st.radio("📌 Navigation", ["Upload", "Explore Data", "Train Model", "Download Model", "Live Prediction"])
    ml_task = st.radio("🤖 Task Type", ["Classification", "Regression"])
    st.info("Upload your data → explore → train → download! & Use Live Prediction")

# Load uploaded data
# Initialize df from session_state or CSV
if "df" not in st.session_state:
    if os.path.exists("sourcedata.csv"):
        st.session_state.df = pd.read_csv("sourcedata.csv", index_col=None)

# Upload Page
if choice == "Upload":
    st.title("📤 Upload Your Dataset")
    file = st.file_uploader("Choose your file (CSV or Excel)", type=["csv", "xlsx"])
    
    if file:
        if file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            st.session_state.df = pd.read_excel(file)
        else:
            st.error("Unsupported file type")
            st.stop()  # stop app if unsupported

        # Save to local CSV
        st.session_state.df.to_csv("sourcedata.csv", index=False)

        st.success("✅ File uploaded and saved!")
        st.dataframe(st.session_state.df.head())
        st.write("Shape:", st.session_state.df.shape)
        st.write("Columns:", list(st.session_state.df.columns))


# EDA
if choice == "Explore Data":
    st.title("📈 Automated Exploratory Data Analysis")
    if "df" in st.session_state:
        with st.spinner("Generating profiling report..."):
            profile = ProfileReport(st.session_state.df, title="Pandas Profiling Report", explorative=True)
            st_profile_report(profile)
    else:
        st.warning("⚠️ Please upload a dataset first.")

# Model Training Section
import datetime
import json
import pickle

if choice == "Train Model":
    st.title("🚀 Train Your Machine Learning Model")
    st.write("DEBUG: Inside Train Model block")  # Debug marker

    # Check if DataFrame is loaded
    if "df" in st.session_state:
        st.write("DEBUG: df is loaded in session_state")  # Debug marker
        st.write(st.session_state.df.head())  # Optional: View a few rows of data

        # Select the target column
        target = st.selectbox("🎯 Select your target column", st.session_state.df.columns)

        # Custom model file name input
        model_filename = st.text_input("📄 Name your model file", "best_model.pkl")

        # Trigger training when button is clicked
        if st.button("Start Training"):
            with st.spinner("Training in progress... this may take a moment."):

                # Run Classification task
                if ml_task == "Classification":
                    st.write("DEBUG: Running classification setup")
                    clf_setup(st.session_state.df, target=target, verbose=False)
                    setup_df = clf_pull()
                    st.info("🔧 ML Experiment Settings:")
                    st.dataframe(setup_df)

                    best_model = clf_compare()
                    compare_df = clf_pull()

                    st.success("✅ Best classification model found!")
                    st.dataframe(compare_df)

                    # Add summary
                    st.subheader("📊 Model Summary")
                    st.write(f"**Best Model:** `{type(best_model).__name__}`")
                    st.write("**Model Parameters:**")
                    st.json(best_model.get_params())

                    # 🎯 Feature Importance
                    st.subheader("🎯 Feature Importance")
                    st.write("Trying to plot feature importance...")
                    try:
                        from pycaret.classification import plot_model
                        plot_model(best_model, plot="feature", display_format="streamlit")
                    except Exception as e:
                        st.warning("⚠️ Could not display feature importance.")
                        st.text(f"Reason: {str(e)}")

                        try:
                            importance = best_model.feature_importances_
                            features = st.session_state.df.drop(columns=target).columns
                            st.write(dict(zip(features, importance)))
                        except:
                            st.info("ℹ️ Model does not support raw feature importance.")

                    # Save actual model with pickle
                    with open(model_filename, "wb") as f:
                        pickle.dump(best_model, f)
                    with open("model_meta.json", "w") as f:
                        json.dump({
                            "model": type(best_model).__name__,
                            "task_type": "Classification",
                            "target_column": target,
                            "trained_on": str(datetime.datetime.now()),
                            "n_features": len(st.session_state.df.columns) - 1,
                            "feature_names": st.session_state.df.drop(columns=[target]).columns.tolist()
                        }, f)

                # Run Regression task
                elif ml_task == "Regression":
                    st.write("DEBUG: Running regression setup")
                    reg_setup(st.session_state.df, target=target, verbose=False)
                    setup_df = reg_pull()
                    st.info("🔧 ML Experiment Settings:")
                    st.dataframe(setup_df)

                    best_model = reg_compare()
                    compare_df = reg_pull()

                    st.success("✅ Best regression model found!")
                    st.dataframe(compare_df)

                    # Add summary
                    st.subheader("📊 Model Summary")
                    st.write(f"**Best Model:** `{type(best_model).__name__}`")
                    st.write("**Model Parameters:**")
                    st.json(best_model.get_params())

                    # 🎯 Feature Importance
                    st.subheader("🎯 Feature Importance")
                    st.write("Trying to plot feature importance...")
                    try:
                        from pycaret.regression import plot_model
                        plot_model(best_model, plot="feature", display_format="streamlit")
                    except Exception as e:
                        st.warning("⚠️ Could not display feature importance.")
                        st.text(f"Reason: {str(e)}")

                        try:
                            importance = best_model.feature_importances_
                            features = st.session_state.df.drop(columns=target).columns
                            st.write(dict(zip(features, importance)))
                        except:
                            st.info("ℹ️ Model does not support raw feature importance.")

                    # Save actual model with pickle
                    with open(model_filename, "wb") as f:
                        pickle.dump(best_model, f)
                    with open("model_meta.json", "w") as f:
                        json.dump({
                            "model": type(best_model).__name__,
                            "task_type": "Regression",
                            "target_column": target,
                            "trained_on": str(datetime.datetime.now()),
                            "n_features": len(st.session_state.df.columns) - 1,
                            "feature_names": st.session_state.df.drop(columns=[target]).columns.tolist()
                        }, f)

            st.balloons()
    else:
        st.warning("⚠️ Please upload and select a dataset first.")

# Model Download Section
if choice == "Download Model":
    st.title("💾 Download Trained Model")
    
    # Model file download
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            st.download_button("📥 Download Model File (.pkl)", f, file_name="best_model.pkl")
    else:
        st.warning("⚠️ No model found. Train one first!")

    # Metadata file download
    if os.path.exists("model_meta.json"):
        with open("model_meta.json", "rb") as meta_file:
            st.download_button("📄 Download Model Metadata (.json)", meta_file, file_name="model_meta.json")
    else:
        st.warning("⚠️ Metadata file not found. Ensure the model was trained.")

    # Optional: Add future report download (e.g. training_report.txt)
    if os.path.exists("training_report.txt"):
        with open("training_report.txt", "rb") as r:
            st.download_button("📝 Download Training Report (optional)", r, file_name="training_report.txt")


# Live Prediction Page
import pickle
import json
import os
from sklearn.base import BaseEstimator

if choice == "Live Prediction":
    st.title("🔮 Live Model Prediction")

    uploaded_model = st.file_uploader("📤 Upload a trained `.pkl` model", type=["pkl"])
    uploaded_metadata = st.file_uploader("📄 Upload the corresponding `model_meta.json`", type=["json"])

    if uploaded_model and uploaded_metadata:
        try:
            # Load model
            loaded_obj = pickle.load(uploaded_model)

            # Ensure the object has predict method
            if isinstance(loaded_obj, BaseEstimator):
                model = loaded_obj
            else:
                raise ValueError("Uploaded file is not a valid model object with 'predict' method.")

            st.success("✅ Model loaded successfully!")

            # Load metadata
            meta = json.load(uploaded_metadata)
            target = meta.get("target_column")

            st.info(f"Using model: `{meta['model']}` for `{meta['task_type']}`")

            # Generate input form dynamically
            st.subheader("✏️ Enter feature values below:")
            features = meta.get("feature_names") or ([col for col in st.session_state.df.columns if col != target] if "df" in st.session_state else [])

            if not features:
                st.warning("⚠️ No feature names found in metadata or dataset. Please retrain model with latest AutoStreamML.")
            else:
                user_input = {}
                for col in features:
                    if "df" in st.session_state and col in st.session_state.df.columns:
                        dtype = st.session_state.df[col].dtype
                        if dtype == 'object':
                            options = st.session_state.df[col].unique().tolist()
                            user_input[col] = st.selectbox(f"{col}", options)
                        else:
                            user_input[col] = st.number_input(f"{col}", value=float(st.session_state.df[col].mean()))
                    else:
                        user_input[col] = st.text_input(f"{col}")

                if st.button("Predict"):
                    input_df = pd.DataFrame([user_input])
                    input_df = input_df.apply(pd.to_numeric, errors='ignore')
                    prediction = model.predict(input_df)
                    st.success(f"🎯 Prediction: `{prediction[0]}`")

        except Exception as e:
            st.error("❌ Failed to load model or predict.")
            st.text(f"Error: {str(e)}")

    elif uploaded_model and not uploaded_metadata:
        st.warning("⚠️ Please upload the metadata file (`model_meta.json`) to continue.")
    elif uploaded_metadata and not uploaded_model:
        st.warning("⚠️ Please upload a trained model `.pkl` file to continue.")
