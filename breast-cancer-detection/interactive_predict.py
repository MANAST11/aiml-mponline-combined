import os
import sys
import random
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt

# Add the current directory to sys.path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.predict import load_saved_model_and_scaler, predict_sample

def print_prediction_result(console, result, actual_diagnosis=None):
    """
    Renders a beautiful prediction panel.
    """
    pred_class = result["diagnosis"]
    prob = result["probability"]
    
    color = "red" if pred_class == "Malignant" else "green"
    
    # Create prediction gauge
    bar_length = 20
    filled = int(round(prob * bar_length))
    bar = "=" * filled + "-" * (bar_length - filled)
    
    content = []
    content.append(f"  [bold]Predicted Diagnosis:[/bold] [{color}]{pred_class}[/{color}]")
    content.append(f"  [bold]Confidence / Probability:[/bold] [yellow]{prob:.2%}[/yellow]")
    content.append(f"  [bold]Confidence Bar:[/bold] [{color}]{bar}[/{color}]")
    
    if actual_diagnosis is not None:
        act_color = "red" if actual_diagnosis == "Malignant" else "green"
        match = "[bold green]Match (Correct)[/bold green]" if pred_class == actual_diagnosis else "[bold red]Mismatch (Incorrect)[/bold red]"
        content.append(f"  [bold]Actual Diagnosis:[/bold] [{act_color}]{actual_diagnosis}[/{act_color}]")
        content.append(f"  [bold]Result Evaluation:[/bold] {match}")
        
    console.print(Panel(
        "\n".join(content),
        title="Prediction Analysis Results",
        border_style=color,
        expand=False
    ))

def display_features_table(console, feature_values, feature_names):
    """
    Displays a table showing the features of the current patient.
    """
    table = Table(title="Patient Tumor Characteristics", border_style="cyan", show_lines=True)
    table.add_column("Feature Group", style="dim cyan")
    table.add_column("Mean Measurement", style="white")
    table.add_column("Standard Error (SE)", style="white")
    table.add_column("Worst Measurement", style="white")
    
    # We group features by their base name (radius, texture, etc.)
    base_features = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
    ]
    
    for base in base_features:
        mean_val = feature_values.get(f"{base}_mean", 0.0)
        se_val = feature_values.get(f"{base}_se", feature_values.get(f"{base}_error", 0.0))
        worst_val = feature_values.get(f"{base}_worst", 0.0)
        
        # Format the name for readability
        display_name = base.replace("_", " ").title()
        
        table.add_row(
            display_name,
            f"{mean_val:.4f}",
            f"{se_val:.4f}",
            f"{worst_val:.4f}"
        )
        
    console.print(table)

def main():
    console = Console()
    console.clear()
    
    console.print(Panel.fit(
        "[bold cyan]Cancer Detection Interactive Predictor[/bold cyan]\n"
        "[dim]Predict tumor malignancy using the trained model in your IDE[/dim]",
        border_style="cyan"
    ))
    
    # Load model and scaler
    try:
        model, scaler = load_saved_model_and_scaler()
        
        # Load feature names
        feature_names_path = os.path.join("models", "feature_names.txt")
        if os.path.exists(feature_names_path):
            with open(feature_names_path, "r") as f:
                feature_names = [line.strip() for line in f.read().splitlines() if line.strip()]
        else:
            # Fallback if text file missing
            feature_names = scaler.feature_names_in_
            
        console.print("[+] Model and scaler loaded successfully from local files.\n", style="green")
    except FileNotFoundError as e:
        console.print(Panel(
            f"[bold red]Error Loading Model[/bold red]\n\n"
            f"{str(e)}\n\n"
            f"Please run [yellow]python main.py[/yellow] to train the model first.",
            border_style="red"
        ))
        return

    # Load dataset for random patient sample selection
    df = load_data()
    
    while True:
        console.print("\n[bold magenta]--- Menu ---[/bold magenta]")
        console.print("[1] Select a random Benign case from dataset")
        console.print("[2] Select a random Malignant case from dataset")
        console.print("[3] Select any random patient case from dataset")
        console.print("[4] Input tumor features manually")
        console.print("[5] Exit")
        
        choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5"], default="3")
        
        if choice == "5":
            console.print("[*] Exiting interactive console. Goodbye!", style="yellow")
            break
            
        # Target variable in our processed df is 'target'
        if choice in ["1", "2", "3"]:
            if choice == "1":
                subset = df[df['target'] == 0]
                label = "Benign"
            elif choice == "2":
                subset = df[df['target'] == 1]
                label = "Malignant"
            else:
                subset = df
                label = "Any"
                
            if len(subset) == 0:
                console.print(f"[red]No {label} samples found in dataset.[/red]")
                continue
                
            # Pick random index
            idx = random.choice(subset.index)
            patient_row = df.loc[idx]
            
            actual_diag = "Malignant" if patient_row['target'] == 1 else "Benign"
            console.print(f"\n[+] Selected Patient index: {idx} (Actual: {actual_diag})", style="cyan")
            
            # Extract features (excluding target)
            features = patient_row.drop('target').to_dict()
            
            display_features_table(console, features, feature_names)
            
            # Predict
            result = predict_sample(model, scaler, features, feature_names)
            print_prediction_result(console, result, actual_diag)
            
        elif choice == "4":
            console.print("\n[bold yellow]Manual Feature Entry Mode[/bold yellow]")
            console.print("Please enter the following values. Press Enter to use the dataset [bold]mean defaults[/bold].\n")
            
            # Calculate means from dataset to use as defaults
            means = df.drop(columns=['target']).mean().to_dict()
            
            # Prompt for major features, auto-fill others
            user_inputs = {}
            
            # We will prompt for the top 6 key features that affect malignancy the most
            key_features_to_prompt = [
                ("radius_mean", "Mean Radius (size)"),
                ("texture_mean", "Mean Texture (gray-scale SD)"),
                ("perimeter_mean", "Mean Perimeter"),
                ("area_mean", "Mean Area"),
                ("smoothness_mean", "Mean Smoothness (local variation in radius)"),
                ("concavity_mean", "Mean Concavity (severity of concave contour portions)")
            ]
            
            for col, description in key_features_to_prompt:
                default_val = means.get(col, 14.0)
                user_val = Prompt.ask(
                    f"Enter [cyan]{description}[/cyan] (default: {default_val:.2f})",
                    default=f"{default_val:.4f}"
                )
                try:
                    user_inputs[col] = float(user_val)
                except ValueError:
                    user_inputs[col] = default_val
            
            # Auto-fill the rest of the 30 features with the overall dataset means
            for col in feature_names:
                if col not in user_inputs:
                    user_inputs[col] = means.get(col, 0.0)
            
            console.print("\n[*] Processing manual entry details...")
            display_features_table(console, user_inputs, feature_names)
            
            # Predict
            result = predict_sample(model, scaler, user_inputs, feature_names)
            print_prediction_result(console, result)

if __name__ == "__main__":
    main()
