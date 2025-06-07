---

# 🚀 AutoStreamML: Automated Machine Learning Web App

**AutoStreamML** is a user-friendly Streamlit web application designed to automate machine learning workflows — from dataset upload and preprocessing to training, evaluation, and model downloading. Whether you're exploring regression or classification tasks, AutoStreamML provides a seamless end-to-end experience with just a few clicks.

> ⚠️ **Live Prediction Feature**: Currently under active development and will be available soon!

---

## 🌟 Key Features

* 📁 Upload and preview CSV datasets
* 🧼 Automatic preprocessing and setup with PyCaret
* 🎯 Support for both **Regression** and **Classification** tasks
* 🤖 Model training with automatic selection of the best-performing model
* 📊 Feature importance visualization (when supported by the model)
* 💾 Download trained model (`.pkl`) and metadata (`model_meta.json`)
* 🧪 (Coming Soon!) Live prediction from user input using trained models

---

## 🔧 Tech Stack

* [Streamlit](https://streamlit.io/)
* [PyCaret](https://pycaret.org/)
* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/)
* [Matplotlib](https://matplotlib.org/)
* [ydata-profiling](https://github.com/ydataai/ydata-profiling)

---

## 🖥️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AutoStreamML.git
cd AutoStreamML
```

### 2. Set up a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you encounter version conflicts, use the pre-tested environment setup:

```bash
pip install numpy==1.26.0 pandas==2.1.3 matplotlib==3.7.3 scipy==1.11.4
pip install pycaret==3.3.2 ydata-profiling==4.6.4 streamlit==1.45.1 streamlit-pandas-profiling==0.1.3
```

### 4. Launch the app

```bash
streamlit run MLAnalysis.py
```


---

## 📌 Status

* ✅ Model upload, training, comparison, and download
* ✅ Feature importance visualization
* 🚧 **Live Prediction UI** — in progress
* 🔜 Future: Dataset versioning, report exports, interactive dashboards

---

## 🤝 Contributing

Interested in improving this project or suggesting a feature? Feel free to open issues or submit a pull request!

---

## 📫 Contact

**Daniels Shashkovs**
Aspiring Machine Learning Engineer


---
