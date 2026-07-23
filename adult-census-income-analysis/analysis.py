import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for plotting without GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
# -------------------------------------------------------------
# Configuration and Setup
# -------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "adult.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
# Helper function to print headers
def print_section_header(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)
def main():
    # -------------------------------------------------------------
    # Task 1: Dataset Understanding
    # -------------------------------------------------------------
    print_section_header("Task 1: Dataset Understanding")
    
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")
    
    # Columns and Types
    print("Column names and data types:")
    print(df.info())
    print()
    
    # Preview Data
    print("First 5 rows:")
    print(df.head())
    print()
    
    # Target Variable Distribution
    print("Target variable ('income') distribution:")
    target_counts = df['income'].value_counts()
    target_pct = df['income'].value_counts(normalize=True) * 100
    for idx, count in target_counts.items():
        print(f"  {idx}: {count} ({target_pct[idx]:.2f}%)")
    print()
    
    # Numerical features summary
    print("Descriptive statistics for numerical columns:")
    print(df.describe())
    print()
    
    # Check for missing values (represented by '?')
    print("Count of '?' (missing values) per column:")
    missing_counts = (df == '?').sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    print()
    # -------------------------------------------------------------
    # Task 2: Data Cleaning
    # -------------------------------------------------------------
    print_section_header("Task 2: Data Cleaning")
    
    # Handle duplicates
    dup_count = df.duplicated().sum()
    print(f"Found {dup_count} duplicate rows. Removing them...")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Shape after duplicate removal: {df.shape}\n")
    
    # Handling missing values '?'
    # Categorical columns with missing values are: workclass, occupation, native.country
    # We will replace '?' with 'Unknown' so we retain all data while explicitly modeling 'Unknown'
    print("Handling missing values represented by '?'...")
    for col in ['workclass', 'occupation', 'native.country']:
        df[col] = df[col].replace('?', 'Unknown')
    
    # Double check '?' counts
    remaining_q = (df == '?').sum().sum()
    print(f"Remaining '?' values in dataset: {remaining_q}\n")
    # -------------------------------------------------------------
    # Task 3: Feature Engineering
    # -------------------------------------------------------------
    print_section_header("Task 3: Feature Engineering")
    
    # 1. Target variable mapping: '<=50K' -> 0, '>50K' -> 1
    print("Mapping target variable 'income' to binary values...")
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    
    # 2. Binary variables: sex (Female -> 0, Male -> 1)
    print("Mapping 'sex' to binary (Female=0, Male=1)...")
    df['sex'] = df['sex'].map({'Female': 0, 'Male': 1})
    
    # 3. Bin native.country into United-States (1) vs Other (0)
    print("Binning 'native.country' into 'native_country_us' (US=1, Other=0)...")
    df['native_country_us'] = (df['native.country'] == 'United-States').astype(int)
    df = df.drop('native.country', axis=1)
    
    # 4. Drop redundant education column (education.num is already the numerical encoded version)
    print("Dropping redundant 'education' column (keeping 'education.num')...")
    df = df.drop('education', axis=1)
    
    # 5. One-Hot Encode other categorical variables
    categorical_cols = ['workclass', 'marital.status', 'occupation', 'relationship', 'race']
    print(f"One-hot encoding categorical variables: {categorical_cols}...")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    print(f"Feature matrix shape after encoding: {df.shape}")
    print(f"Columns: {list(df.columns[:10])} ... [showing first 10 columns]\n")
    
    # 6. Train-test split (80% Train, 20% Test)
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples\n")
    
    # 7. Scale Numerical Features
    numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    print(f"Scaling numerical columns using StandardScaler: {numerical_cols}...")
    scaler = StandardScaler()
    
    # Copy dataframes to avoid setting with copy warnings
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    print("Scaling complete.\n")
    # -------------------------------------------------------------
    # Task 4: Model Building
    # -------------------------------------------------------------
    print_section_header("Task 4: Model Building")
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    # For SVM, to prevent slow training on CPU, we will train it on a random subset
    # of 10,000 samples from the training set, which is representative enough.
    # KNN prediction can also be sped up if needed, but 5k is very fast.
    svm_sample_size = 10000
    np.random.seed(42)
    svm_indices = np.random.choice(len(X_train_scaled), size=svm_sample_size, replace=False)
    X_train_svm = X_train_scaled.iloc[svm_indices]
    y_train_svm = y_train.iloc[svm_indices]
    
    results = {}
    
    # Train and predict for each model
    for name, model in models.items():
        print(f"Training {name}...")
        if name == "SVM":
            print(f"  (Note: SVM is trained on a representative sample of {svm_sample_size} training instances to ensure fast CPU run times.)")
            model.fit(X_train_svm, y_train_svm)
        else:
            model.fit(X_train_scaled, y_train)
            
        print(f"Evaluating {name}...")
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Save predictions and probabilities for evaluation
        results[name] = {
            "y_pred": y_pred,
            "y_prob": y_prob,
            "model_obj": model
        }
    print("Model training and prediction complete.\n")
    # -------------------------------------------------------------
    # Task 5: Performance Evaluation
    # -------------------------------------------------------------
    print_section_header("Task 5: Performance Evaluation")
    
    metrics_summary = []
    
    # Setup visualization style
    sns.set_theme(style="whitegrid")
    
    # Create figure for combined ROC Curve
    plt.figure(figsize=(10, 8))
    
    for name in models.keys():
        y_pred = results[name]["y_pred"]
        y_prob = results[name]["y_prob"]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        metrics_summary.append({
            "Algorithm": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })
        
        # Plot ROC curve for the combined plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", lw=2)
        
        # Plot Individual Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['<=50K', '>50K'],
            yticklabels=['<=50K', '>50K'],
            cbar=False, annot_kws={"size": 14}
        )
        plt.title(f"Confusion Matrix - {name}", fontsize=14, fontweight='bold', pad=15)
        plt.ylabel("Actual Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"Saved confusion matrix plot for {name} to results/")
        
    # Finalize and save combined ROC curve plot
    plt.figure(1)  # Focus back on the ROC curve figure
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, "roc_curves_comparison.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print("Saved ROC curves comparison plot to results/\n")
    
    # Create DataFrame for results comparison
    metrics_df = pd.DataFrame(metrics_summary)
    
    # Save metrics to CSV
    metrics_csv_path = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics summary table to {metrics_csv_path}\n")
    
    # Print results summary table
    print("Performance Comparison Table:")
    print(metrics_df.to_string(index=False, formatters={
        "Accuracy": "{:.4%}".format,
        "Precision": "{:.4%}".format,
        "Recall": "{:.4%}".format,
        "F1-Score": "{:.4%}".format,
        "ROC-AUC": "{:.4%}".format
    }))
    print()
if __name__ == "__main__":
    main()
