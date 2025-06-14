# MLAnalysis.py

import streamlit as st
import pandas as pd
import os
import datetime
import json

# Profiling and ML
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

# ML models and tools
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
import joblib
import pydeck as pdk

# Set page config
st.set_page_config(page_title="AutoStreamML", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-03.png", width=250)
    st.title("AutoStreamML")
    st.markdown("📊 AutoML Dashboard for Classification, Regression, and Clustering")
    choice = st.radio("📌 Navigation", ["Upload", "Explore Data", "Train Model", "Download Model", "Live Prediction"])
    st.info("Upload your data → explore → train → download! & Use Live Prediction")

# Load uploaded data
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
            st.stop()

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

# Utility: Auto Cleaning Function

def auto_clean_data(df):
    log = []

    # Drop duplicate rows
    before = df.shape[0]
    df = df.drop_duplicates()
    if df.shape[0] < before:
        log.append(f"Removed {before - df.shape[0]} duplicate rows.")

    # Drop constant columns (only one unique value)
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        log.append(f"Dropped constant columns: {', '.join(constant_cols)}")

    # Optionally drop typical ID columns by naming convention
    id_cols = [col for col in df.columns if col.lower() in ['id', 'index', 'code']]
    if id_cols:
        df = df.drop(columns=id_cols)
        log.append(f"Dropped typical ID columns: {', '.join(id_cols)}")

    # Drop columns with >50% missing
    high_nulls = [col for col in df.columns if df[col].isnull().mean() > 0.5]
    if high_nulls:
        df = df.drop(columns=high_nulls)
        log.append(f"Dropped columns with >50% missing values: {', '.join(high_nulls)}")

    return df, log

# Train Model
if choice == "Train Model":
    st.title("🚀 Train Your Machine Learning Model")

    if "df" in st.session_state:
        df = st.session_state.df.copy()

        # Auto clean the data
        df, cleaning_log = auto_clean_data(df)
        if cleaning_log:
            st.info("\n".join(cleaning_log))

        if df.isnull().values.any():
            st.warning("⚠️ Your dataset contains missing values. Imputation will be applied automatically based on feature type.")

        model_choice = st.selectbox("🧠 Choose model", [
            "RandomForest", "Linear/Logistic Regression", "KMeans (Clustering)"
        ])

        if model_choice == "KMeans (Clustering)":
            ml_task = "Clustering"
            X = df
            y = None
        else:
            target = st.selectbox("🎯 Select your target column", df.columns)
            X = df.drop(columns=[target])
            y = df[target]
            ml_task = "Classification" if y.nunique() <= 10 else "Regression"

        model_filename = st.text_input("📄 Name your model file", "best_model")

        if st.button("Start Training"):
            try:
                numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

                numeric_transformer = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
                categorical_transformer = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

                preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, numeric_features),
                        ("cat", categorical_transformer, categorical_features)
                    ])

                if model_choice == "RandomForest":
                    model_core = RandomForestClassifier() if ml_task == "Classification" else RandomForestRegressor()
                elif model_choice == "Linear/Logistic Regression":
                    model_core = LogisticRegression(max_iter=1000) if ml_task == "Classification" else LinearRegression()
                elif model_choice == "KMeans (Clustering)":
                    model_core = KMeans(n_clusters=3)

                model = make_pipeline(preprocessor, model_core)

                if ml_task != "Clustering":
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    if ml_task == "Classification":
                        score = accuracy_score(y_test, predictions)
                        st.success(f"✅ Model trained. Accuracy: {score:.4f}")
                    else:
                        mse = mean_squared_error(y_test, predictions)
                        r2 = r2_score(y_test, predictions)
                        mae = mean_absolute_error(y_test, predictions)
                        st.success(f"✅ Model trained. MSE: {mse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}")
                else:
                    model.fit(X)
                    st.success("✅ Clustering model trained successfully!")

                joblib.dump(model, f"{model_filename}.pkl")

                meta = {
                    "model": model_choice,
                    "task_type": ml_task,
                    "target_column": target if ml_task != "Clustering" else None,
                    "trained_on": str(datetime.datetime.now()),
                    "n_features": X.shape[1],
                    "feature_names": X.columns.tolist(),
                    "model_filename": model_filename,
                    "value_ranges": {col: [float(X[col].min()), float(X[col].max())] for col in numeric_features},
                    "feature_explanations": {
                        "longitude": "Geographical longitude of the block",
                        "latitude": "Geographical latitude of the block",
                        "housing_median_age": "Median age of houses in the block",
                        "total_rooms": "Total number of rooms (all types) in the block",
                        "total_bedrooms": "Total number of bedrooms in the block",
                        "population": "Total number of people residing in the block",
                        "households": "Number of households in the block",
                        "median_income": "Median income of households in the block (scaled in tens of thousands)",
                        "ocean_proximity": "How close the area is to the ocean"
                    }
                }
                with open("model_meta.json", "w") as f:
                    json.dump(meta, f)

                with open("training_report.txt", "w") as r:
                    r.write(f"Training Report\n")
                    r.write(f"=================\n")
                    r.write(f"Date: {meta['trained_on']}\n")
                    r.write(f"Task: {meta['task_type']}\n")
                    r.write(f"Model: {meta['model']}\n")
                    r.write(f"Target: {meta['target_column']}\n")
                    r.write(f"Features Used: {meta['feature_names']}\n")
                    r.write(f"Feature Count: {meta['n_features']}\n")
                    if ml_task != "Clustering":
                        if ml_task == "Classification":
                            r.write(f"Accuracy: {score:.4f}\n")
                        else:
                            r.write(f"MSE: {mse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}\n")

                st.balloons()

            except Exception as e:
                st.error("❌ Training failed: " + str(e))

    else:
        st.warning("⚠️ Please upload and select a dataset first.")

# Download Model Section
if choice == "Download Model":
    st.title("💾 Download Trained Model")

    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            st.download_button("📥 Download Model File (.pkl)", f, file_name="best_model.pkl")
    else:
        st.warning("⚠️ No model found. Train one first!")

    if os.path.exists("model_meta.json"):
        with open("model_meta.json", "rb") as meta_file:
            st.download_button("📄 Download Model Metadata (.json)", meta_file, file_name="model_meta.json")
    else:
        st.warning("⚠️ Metadata file not found. Ensure the model was trained.")

    if os.path.exists("training_report.txt"):
        with open("training_report.txt", "rb") as r:
            st.download_button("📝 Download Training Report (optional)", r, file_name="training_report.txt")
    else:
        st.warning("⚠️ Training report not found. Re-train your model to regenerate it.")

# Live Prediction Section
if choice == "Live Prediction":
    st.title("🔮 Live Model Prediction")
    model_file = st.file_uploader("📤 Upload a trained model (.pkl)", type=["pkl"])
    meta_file = st.file_uploader("📄 Upload the corresponding model metadata (.json)", type=["json"])

    if model_file and meta_file:
        try:
            # File Handling: Save uploaded files to disk
            with open("uploaded_model.pkl", "wb") as f:
                f.write(model_file.read())
            with open("uploaded_metadata.json", "w") as f:
                # Decode JSON file content with UTF-8
                try:
                    content = meta_file.read().decode("utf-8")
                    f.write(content)
                except UnicodeDecodeError:
                    st.error("Failed to decode metadata file. Ensure it’s a valid JSON file.")
                    st.stop()

            # Load Model and Metadata
            try:
                model = joblib.load("uploaded_model.pkl")
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                st.stop()

            try:
                with open("uploaded_metadata.json") as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON in metadata file: {str(e)}")
                st.stop()

            # Extract metadata
            features = metadata.get("feature_names", [])
            explanations = metadata.get("feature_explanations", {})
            value_ranges = metadata.get("value_ranges", {})

            if not features:
                st.warning("No feature names found in metadata. Please check the JSON file.")
                st.stop()

            # Collect user inputs via Streamlit
            user_input = {}
            st.subheader("✏️ Input Features")

            for feat in features:
                exp = explanations.get(feat, "")
                if feat == "ocean_proximity":
                    user_input[feat] = st.selectbox(
                        f"{feat} - {exp}",
                        ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
                    )
                elif feat in value_ranges:
                    try:
                        min_val, max_val = value_ranges[feat]
                        user_input[feat] = st.slider(
                            f"{feat} - {exp}",
                            float(min_val),
                            float(max_val),
                            float((min_val + max_val) / 2)
                        )
                    except (ValueError, TypeError) as e:
                        st.warning(f"Slider not available for {feat}: {str(e)}. Using text input instead.")
                        user_input[feat] = st.text_input(f"{feat} - {exp}")
                else:
                    user_input[feat] = st.text_input(f"{feat} - {exp}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# 🗺️ Optional Location Map (if lat/lon present)
st.subheader("🗺️ Location Map (Latitude & Longitude)")

if "latitude" in user_input and "longitude" in user_input:
    try:
        lat = float(user_input["latitude"])
        lon = float(user_input["longitude"])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=10, pitch=50),
                layers=[
                    pdk.Layer("ScatterplotLayer", data=pd.DataFrame({"lat": [lat], "lon": [lon]}),
                              get_position='[lon, lat]', get_color='[200, 30, 0, 160]', get_radius=200)
                ]
            ))
        else:
            st.warning("Coordinates are out of bounds. Latitude must be between -90 and 90, longitude between -180 and 180.")
    except ValueError:
        st.warning("Latitude and Longitude must be valid numbers.")

# ✅ Prediction button
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_input])
        input_df = input_df.apply(pd.to_numeric, errors='ignore')
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 Prediction: `{prediction}`")
    except Exception as e:
        st.error("❌ Prediction failed.")
        st.text(str(e))
