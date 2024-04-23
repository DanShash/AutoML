```markdown
# AutoStreamML

AutoStreamML is a Streamlit application that allows you to build an automated ML pipeline for data analysis and model training.

## Description

This application provides a user-friendly interface to upload a dataset, perform automated exploratory data analysis (EDA) using Pandas profiling, and train machine learning models for classification tasks. It offers insights into the dataset's characteristics and allows users to select the target variable for model training.

## How to Run

To run this application locally, follow these steps:

1. Download or clone this repository to your local machine.
2. Navigate to the project directory in your terminal.
3. Install the required Python packages by running:
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit app by executing the following command:
   ```
   streamlit run MLAnalysis.py
   ```
5. The application will open in your default web browser at `http://localhost:8501`.

## Usage

- **Upload:** Upload your dataset for analysis.
- **Analytics:** Perform automated exploratory data analysis.
- **ML-training:** Train machine learning models on the dataset.
- **Download:** Download the best-trained model for future use.

**Note:** Larger datasets may take longer to analyze and train models.

## Example

- Dataset: "UploadData.csv"
- Target variable: "CUSTOMER_REVIEWS"
- Classification model accuracy: 0.63

By analyzing the dataset and training various models, you can determine the best model for your business goals and download it for future use.

**Logs:** After training, a "logs.txt" file will appear in the application folder.

## Screenshots

Screenshots of the application interface can be found in the `screenshots` directory.

Feel free to explore the application and experiment with different datasets and models!
```
