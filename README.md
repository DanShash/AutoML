# 🚀 AutoStreamML: Automated Machine Learning Web App

**AutoStreamML** is a user-friendly Streamlit web application designed to automate machine learning workflows—from dataset upload and cleaning to model training, evaluation, live prediction, and downloading. Whether you're exploring regression, classification, or clustering tasks, AutoStreamML provides a seamless end-to-end experience with just a few clicks.

> ✅ **Live Prediction Feature**: Now available with interactive map support and input explanations!

---

## 🌟 Key Features

* 📁 Upload and preview CSV/XLSX datasets
* 🧼 Auto-cleaning: handles duplicates, missing values, constant columns, and noisy data
* 📊 Interactive EDA profiling using ydata-profiling
* 🎯 Choose between **Regression**, **Classification**, and **Clustering** tasks
* 🧠 Train models using **Random Forest**, **Linear/Logistic Regression**, or **KMeans**
* 🧪 Live prediction with:
  - Sliders & dropdowns
  - Contextual feature explanations
  - Interactive geographic map using latitude & longitude
* 📄 Downloadable outputs:
  - Trained model (`.pkl`)
  - Metadata JSON (`model_meta.json`)
  - Summary report with training details & metrics

---

## 🔧 Tech Stack

* [Streamlit](https://streamlit.io/)
* [scikit-learn](https://scikit-learn.org/)
* [pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [PyDeck](https://deckgl.readthedocs.io/en/latest/)
* [ydata-profiling](https://github.com/ydataai/ydata-profiling)
* [Joblib](https://joblib.readthedocs.io/en/latest/)

---

## 🖥️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AutoStreamML.git
cd AutoStreamML
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you encounter version conflicts, try this:

```bash
pip install numpy==1.26.0 pandas==2.2.2 matplotlib==3.7.3 scipy==1.11.4
pip install scikit-learn==1.4.2 ydata-profiling==4.6.4 streamlit==1.45.1 streamlit-pandas-profiling==0.1.3 pydeck==0.8.0 joblib
```

### 4. Launch the app

```bash
streamlit run MLAnalysis.py
```

---

## 📌 Project Status

* ✅ Data upload, preprocessing, and auto-cleaning
* ✅ Exploratory Data Analysis (EDA)
* ✅ Model selection, training, and evaluation
* ✅ Download of trained models, metadata, and reports
* ✅ Fully functional live prediction with visual guidance
* ✅ Interactive map integration (using PyDeck)
* 🚧 Planned: More model types, support for multi-target prediction, and performance tuning UI

---

## 🤝 Contributing

Interested in improving this project or suggesting a feature? Feel free to open issues or submit a pull request!

---

## 📫 Contact

**Daniels Shashkovs**  
Aspiring Machine Learning Engineer  
[GitHub Profile](https://github.com/DanShash)

---
