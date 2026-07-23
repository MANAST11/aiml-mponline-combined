import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def find_dataset():
    """Look for the dataset CSV in standard locations."""
    possible_paths = [
        "Cancer_Data.csv",
        "data.csv",
        "data/Cancer_Data.csv",
        "data/data.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def load_data(filepath=None):
    """
    Loads the breast cancer dataset.
    If filepath is provided or found, loads from CSV.
    Otherwise, loads the identical dataset from sklearn as a fallback.
    """
    if filepath is None:
        filepath = find_dataset()
        
    if filepath and os.path.exists(filepath):
        print(f"[*] Loading dataset from CSV file: {filepath}")
        df = pd.read_csv(filepath)
        
        # Clean up columns: drop 'Unnamed: 32' or similar empty columns
        if 'Unnamed: 32' in df.columns:
            df = df.drop(columns=['Unnamed: 32'])
        
        # Drop ID if it exists
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            
        # Encode target variable: diagnosis 'M' -> 1 (Malignant), 'B' -> 0 (Benign)
        if 'diagnosis' in df.columns:
            df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
            df = df.drop(columns=['diagnosis'])
        
        # Ensure target column is named 'target'
        return df
    else:
        print("[!] CSV dataset not found locally. Loading standard breast cancer dataset from sklearn as a fallback.")
        # Load from sklearn and rename columns to match the Kaggle dataset style
        cancer = load_breast_cancer(as_frame=True)
        df = cancer.frame.copy()
        
        # In sklearn, target: 0 = malignant, 1 = benign.
        # We map target: 0 -> 1 (Malignant), 1 -> 0 (Benign) to align with Kaggle (M -> 1, B -> 0)
        df['target'] = df['target'].map({0: 1, 1: 0})
        
        # Rename sklearn columns (e.g. 'mean radius' -> 'radius_mean') to match Kaggle format
        rename_dict = {}
        for col in df.columns:
            if col == 'target':
                continue
            # Handle standard error / error
            new_col = col
            if 'error' in col:
                new_col = col.replace(' error', '_se')
            elif 'mean' in col:
                new_col = col.replace('mean ', '') + '_mean'
            elif 'worst' in col:
                new_col = col.replace('worst ', '') + '_worst'
            
            # Replace spaces with underscores
            new_col = new_col.replace(' ', '_')
            rename_dict[col] = new_col
            
        df = df.rename(columns=rename_dict)
        return df

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Splits features and target, performs train-test split,
    and scales features.
    """
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Split (stratified to ensure identical class distributions in train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Return features as pandas DataFrames/Series to preserve column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler
