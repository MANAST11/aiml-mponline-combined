import os
import joblib
import pandas as pd
import numpy as np

def load_saved_model_and_scaler(model_dir="models"):
    """
    Loads the trained model and scaler from the specified directory.
    """
    model_path = os.path.join(model_dir, "breast_cancer_model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "Trained model or scaler files not found. "
            "Please run 'main.py' first to train and save the model."
        )
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def predict_sample(model, scaler, features_dict, feature_names):
    """
    Predicts if a single sample is malignant or benign.
    
    Parameters:
    - model: trained classifier
    - scaler: trained StandardScaler
    - features_dict: dictionary containing feature names and values
    - feature_names: list of all feature names expected by the scaler/model in order
    """
    # Build list in correct order
    input_data = []
    for col in feature_names:
        input_data.append(features_dict[col])
        
    # Convert to DataFrame to retain names for warnings / consistency
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
    
    # Predict
    prediction = model.predict(input_scaled_df)[0]
    
    # Get probabilities if supported
    probability = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_scaled_df)[0]
        # class 1 corresponds to Malignant, class 0 to Benign
        probability = probs[1] if prediction == 1 else probs[0]
        
    diagnosis = "Malignant" if prediction == 1 else "Benign"
    
    return {
        "prediction": prediction,
        "diagnosis": diagnosis,
        "probability": probability
    }
