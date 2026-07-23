# Adult Census Income Classification Project
This project implements a complete machine learning workflow to classify individuals' income levels based on demographic and employment data. The target variable is whether an individual's income exceeds $50,000 per year (`>50K` mapped to `1`) or not (`<=50K` mapped to `0`).
We build, evaluate, and compare 5 machine learning algorithms:
1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **K-Nearest Neighbors (KNN)**
5. **Support Vector Machine (SVM)**
---
## Project Structure
```
adult_census_project/
│
├── data/
│   └── adult.csv                      # Raw dataset (Downloaded automatically)
│
├── results/
│   ├── evaluation_metrics.csv         # Table of model metrics
│   ├── roc_curves_comparison.png      # Combined ROC Curve plot for all 5 models
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_knn.png
│   └── confusion_matrix_svm.png
│
├── src/
│   ├── download_data.py               # Data acquisition script
│   └── analysis.py                    # Data cleaning, feature engineering, model training & evaluation
│
└── README.md                          # Project Summary & Evaluation Report (This file)
```
---
## How to Run the Project
### Prerequisites
Make sure you have Python installed, along with the required libraries. If needed, install them globally or in your virtual environment:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
### Steps to Run
1. **Download the Dataset**:
   Run the download script to fetch the dataset and set up the directories.
   ```bash
   python src/download_data.py
   ```
2. **Run the Machine Learning Pipeline**:
   Run the analysis script to execute data cleaning, feature engineering, train the models, and generate evaluation files under `results/`.
   ```bash
   python src/analysis.py
   ```
---
## Detailed Task Breakdown & Analysis
### Task 1: Dataset Understanding
*   **Dimensions**: The raw dataset contains **32,561 rows** and **15 columns**.
*   **Target Variable Class Balance**:
    *   `<=50K`: **24,720 (75.92%)**
    *   `>50K`: **7,841 (24.08%)**
    *   *Observation*: The dataset exhibits a significant class imbalance (approx. 3:1 ratio), making metrics like F1-Score and ROC-AUC much more informative than Accuracy alone.
*   **Data Types**:
    *   *Numerical (6 columns)*: `age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`, `hours.per.week`.
    *   *Categorical (9 columns)*: `workclass`, `education`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `native.country`, `income`.
*   **Missing Values**:
    Missing values are represented implicitly as `"?"` in the raw text:
    *   `workclass`: **1,836 (5.64%)** missing
    *   `occupation`: **1,843 (5.66%)** missing
    *   `native.country`: **583 (1.79%)** missing
### Task 2: Data Cleaning
1. **Duplicate Removal**: Identified and dropped **24 duplicate rows** to prevent data leakage and overfitting.
2. **Missing Values**:
   - Rather than dropping rows with missing values (which would lose ~7.4% of the data), we replaced `"?"` with `"Unknown"`. 
   - This retains the full dataset size (**32,537 rows** remaining after deduplication) and treats the lack of information in `workclass`, `occupation`, and `native.country` as an explicit, predictive category.
### Task 3: Feature Engineering
1. **Target Variable Mapping**: Mapped `income` directly to binary values (`<=50K` $\rightarrow$ `0`, `>50K` $\rightarrow$ `1`).
2. **Binary Variable Mapping**: Mapped `sex` directly to binary values (`Female` $\rightarrow$ `0`, `Male` $\rightarrow$ `1`).
3. **Dimensionality Reduction (native.country)**: 
   - Because `native.country` is highly skewed (89.6% are from the United States), one-hot encoding all 41 categories would create sparse, noisy columns.
   - We binned the column into `native_country_us` (`1` if United-States, `0` otherwise).
4. **Redundancy Removal (education)**:
   - Dropped the categorical `education` column because `education.num` serves as its exact, clean pre-ordinal representation.
5. **One-Hot Encoding**: Used `pd.get_dummies` with `drop_first=True` on remaining categorical variables (`workclass`, `marital.status`, `occupation`, `relationship`, `race`) to convert them to numerical format. This expanded our features to a total of 45 predictors.
6. **Train-Test Split**: Performed an **80-20 stratified split** (train size: 26,029, test size: 6,508) to maintain the target class distribution.
7. **Feature Scaling**: Standardized all continuous columns (`age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`, `hours.per.week`) using `StandardScaler` fitted on the training set to prevent data leakage.
### Task 4: Model Building
We trained the following five algorithms:
*   **Logistic Regression**: Fits a linear boundary; very fast and highly interpretable.
*   **Decision Tree**: Non-linear model; limited to `max_depth=10` to avoid overfitting.
*   **Random Forest**: Ensemble of 100 Decision Trees; handles non-linearities and reduces variance.
*   **K-Nearest Neighbors (KNN)**: Distance-based classifier; configured with $k=5$.
*   **Support Vector Machine (SVM)**: Fits an optimal margin hyperplane. To ensure reasonable execution speeds on CPU, the SVM model was trained on a representative, stratified random subset of **10,000 training samples** and evaluated on the full test set.
---
## Task 5: Performance Evaluation
### Summary Performance Comparison Table
|
 Algorithm 
|
 Accuracy 
|
 Precision 
|
 Recall 
|
 F1-Score 
|
 ROC-AUC 
|
|
:---
|
:---:
|
:---:
|
:---:
|
:---:
|
:---:
|
|
**
Logistic Regression
**
|
 85.1721% 
|
 73.5731% 
|
 60.0128% 
|
 66.1047% 
|
**
90.1602%
**
|
|
**
Decision Tree
**
|
 84.8341% 
|
 71.0660% 
|
**
62.5000%
**
|
 66.5083% 
|
 88.5890% 
|
|
**
Random Forest
**
|
 85.2182% 
|
 73.0945% 
|
 61.1607% 
|
**
66.5972%
**
|
 89.7431% 
|
|
**
KNN
**
|
 82.7597% 
|
 65.9971% 
|
 58.6735% 
|
 62.1202% 
|
 85.1024% 
|
|
**
SVM
**
|
**
85.3719%
**
|
**
76.0135%
**
|
 57.3980% 
|
 65.4070% 
|
 89.8327% 
|
*All metrics are evaluated on the held-out 20% test set (6,508 samples).*
### Key Insights and Discussion
1. **Overall Performance**:
   * **Accuracy**: **SVM** achieved the highest overall accuracy of **85.37%**, followed closely by **Random Forest (85.22%)** and **Logistic Regression (85.17%)**. 
   * **ROC-AUC**: **Logistic Regression** achieved the best ROC-AUC score of **90.16%**, indicating excellent class separation. SVM (89.83%) and Random Forest (89.74%) were highly competitive.
2. **Precision vs. Recall Trade-offs**:
   * **SVM** had the highest **Precision (76.01%)**. This means when SVM predicts an individual earns $>50K$, it is correct **76.01%** of the time. However, it had a lower **Recall (57.40%)**, missing some high earners.
   * **Decision Tree** achieved the highest **Recall (62.50%)**, identifying more individuals earning $>50K$ but at the cost of a lower **Precision (71.07%)** (more false positives).
   * **Random Forest** provides the most balanced trade-off, achieving the highest **F1-Score (66.60%)**.
3. **Algorithm Comparisons**:
   * **Logistic Regression** performed exceptionally well, showing that after standardization and one-hot encoding, the relationship between features and income is heavily linear or can be captured well by a linear decision boundary.
   * **KNN** performed the worst across all metrics (82.76% accuracy, 85.10% ROC-AUC), as it struggles with high-dimensional spaces where irrelevant dimensions dilute distance measures (known as the *curse of dimensionality*).
   * **Random Forest** beats **Decision Tree** in accuracy, precision, and ROC-AUC, demonstrating the power of bagging and ensembles in reducing variance and overfitting.
### Visualization Artifacts
The following plots are generated and saved in the `results/` folder to visually analyze the models:
*   `roc_curves_comparison.png`: A single plot showing the ROC curves of all five models together. This visually highlights Logistic Regression and SVM's superior class separation capabilities.
*   `confusion_matrix_<algorithm>.png`: Heatmaps depicting the True Positives, True Negatives, False Positives, and False Negatives for each model.
