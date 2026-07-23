import os
import sys
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add the current directory to sys.path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data, preprocess_data
from src.model import train_models, evaluate_models, save_best_model

def main():
    console = Console()
    console.print(Panel.fit(
        "[bold cyan]Breast Cancer Detection Pipeline[/bold cyan]\n"
        "[dim]Trains, evaluates and saves machine learning models for diagnosis[/dim]",
        border_style="cyan"
    ))
    
    # 1. Load Data
    console.print("[*] loading dataset...", style="yellow")
    df = load_data()
    
    console.print(f"[+] Dataset loaded successfully. Shape: {df.shape}", style="green")
    
    # Print class distribution
    class_counts = df['target'].value_counts()
    malignant_count = class_counts.get(1, 0)
    benign_count = class_counts.get(0, 0)
    total = len(df)
    
    console.print(f"    - Malignant cases (1): {malignant_count} ({malignant_count/total:.1%})", style="red")
    console.print(f"    - Benign cases (0): {benign_count} ({benign_count/total:.1%})", style="green")
    
    # 2. Preprocess Data
    console.print("\n[*] Preprocessing and splitting data (80% train, 20% test)...", style="yellow")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    console.print(f"[+] Preprocessed features count: {X_train.shape[1]}", style="green")
    
    # Save the feature names for inference
    feature_names_path = os.path.join("models", "feature_names.txt")
    os.makedirs("models", exist_ok=True)
    with open(feature_names_path, "w") as f:
        f.write("\n".join(X_train.columns))
    console.print(f"[+] Saved feature names to: {feature_names_path}", style="green")
    
    # 3. Train Models
    console.print("\n[*] Training classifiers...", style="yellow")
    trained_models = train_models(X_train, y_train)
    
    # 4. Evaluate Models
    console.print("\n[*] Evaluating models on test set...", style="yellow")
    results = evaluate_models(trained_models, X_test, y_test)
    
    # Display comparison table
    table = Table(title="Model Evaluation Summary", border_style="cyan")
    table.add_column("Classifier Name", style="bold white")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Precision", justify="right", style="green")
    table.add_column("Recall (Sensitivity)", justify="right", style="green")
    table.add_column("F1-Score", justify="right", style="bold green")
    
    for name, metrics in results.items():
        table.add_row(
            name,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}"
        )
        
    console.print(table)
    
    # Print classification reports
    for name, metrics in results.items():
        console.print(Panel(
            metrics['classification_report'],
            title=f"Classification Report: {name}",
            border_style="magenta"
        ))
        
    # 5. Save the best model
    best_name, model_path, scaler_path = save_best_model(results, scaler)
    
    console.print(Panel.fit(
        f"[bold green][+] Training Pipeline Completed Successfully![/bold green]\n"
        f"Best Model: [yellow]{best_name}[/yellow]\n"
        f"Saved model path: [cyan]{model_path}[/cyan]\n"
        f"Saved scaler path: [cyan]{scaler_path}[/cyan]",
        title="Status",
        border_style="green"
    ))

if __name__ == "__main__":
    main()
