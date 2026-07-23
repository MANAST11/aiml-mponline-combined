import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def train_models(X_train, y_train, random_state=42):
    """
    Trains multiple machine learning models and returns them in a dictionary.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Support Vector Classifier (SVC)": SVC(probability=True, random_state=random_state)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"[*] Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluates trained models on the test set and prints detailed reports.
    Returns metrics dict.
    """
    evaluation_results = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        evaluation_results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])
        }
        
    return evaluation_results

def save_best_model(evaluation_results, scaler, output_dir="models"):
    """
    Selects the best model based on F1-score and saves it along with the scaler.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    best_name = None
    best_f1 = -1
    best_model = None
    
    for name, results in evaluation_results.items():
        if results["f1_score"] > best_f1:
            best_f1 = results["f1_score"]
            best_name = name
            best_model = results["model"]
            
    print(f"\n[+] Best Model Selected: {best_name} with F1-score: {best_f1:.4f}")
    
    model_path = os.path.join(output_dir, "breast_cancer_model.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    
    # Save model and scaler
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"[+] Saved model to: {model_path}")
    print(f"[+] Saved scaler to: {scaler_path}")
    
    return best_name, model_path, scaler_path
