# Breast Cancer Detection & Classification Project

This is a Python-based Machine Learning project for predicting whether a breast tumor is **Malignant** or **Benign** based on cell nuclei diagnostic measurements. The dataset used is the classic Breast Cancer Wisconsin (Diagnostic) dataset, which is hosted on Kaggle: [Cancer Data by Erdem Taha](https://www.kaggle.com/datasets/erdemtaha/cancer-data).

The project is designed to run entirely in your local IDE (such as PyCharm, VS Code, or standard command prompt) and uses command-line interfaces with rich layouts.

---

## Project Structure

```
cancer_detection/
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Handles dataset loading, cleaning, and preprocessing
│   ├── model.py             # Handles training, evaluation, and saving pipelines
│   └── predict.py           # Handles prediction on individual samples
├── main.py                  # Entry point to train and save models
├── interactive_predict.py   # IDE-runnable interactive console for predictions
├── requirements.txt         # Package dependencies
└── README.md                # Project documentation (this file)
```

---

## Requirements and Installation

All necessary libraries are already pre-installed in your environment. If you want to set it up elsewhere, you can install the dependencies via:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. (Optional) Provide the Kaggle CSV
You can download the dataset from Kaggle and place the CSV file in the root directory or inside a `data/` folder.
*   The script automatically looks for file names: `Cancer_Data.csv` or `data.csv`.
*   **Automatic Fallback:** If you do not download the CSV, the code will automatically load the identical dataset from `scikit-learn`'s built-in repository, allowing you to run the project instantly without any manual downloads!

### 2. Run the Training Pipeline
Run `main.py` in your terminal or IDE. This script will:
*   Load and clean the dataset.
*   Split the data (80% training, 20% testing) and apply feature scaling.
*   Train three classifiers: Logistic Regression, Random Forest, and Support Vector Classifier (SVC).
*   Print a clean evaluation table comparing their Accuracy, Precision, Recall, and F1-Score.
*   Save the best performing model (based on F1-score) and the scaler to the `models/` directory.

To run the script:
```bash
python main.py
```

### 3. Run the Interactive Predictor
Once you have trained the model, run `interactive_predict.py` in your IDE. This CLI dashboard lets you:
*   Option 1: Grab a random **Benign** patient case from the dataset, see its cell values, and inspect the prediction probability.
*   Option 2: Grab a random **Malignant** patient case from the dataset, see its cell values, and inspect the prediction probability.
*   Option 3: Grab any random patient case from the dataset.
*   Option 4: Manually enter key features (e.g. mean radius, texture, area) and get an instant diagnosis classification with a visual confidence bar.

To run the CLI:
```bash
python interactive_predict.py
```

---

## Machine Learning Details

### Evaluation Strategy
For medical applications, **Recall (Sensitivity)** and **F1-Score** are crucial:
*   **Recall (Sensitivity):** Minimizes False Negatives (cases where cancer is present but predicted as benign).
*   **Precision:** Minimizes False Positives (cases where benign tumors are flagged as malignant).
*   **F1-Score:** Harmonic mean of precision and recall, serving as our primary metric for selecting the best model.
