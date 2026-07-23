import os
import argparse
from ultralytics import YOLO

def train_model(epochs, batch):
    yaml_path = os.path.abspath(os.path.join('demo_dataset', 'data.yaml'))
    if not os.path.exists(yaml_path):
        print(f"Error: YOLOv8 configuration file not found at: {yaml_path}")
        print("Please run 'python prepare_demo_dataset.py' first to build the dataset.")
        return
        
    print("Loading pre-trained YOLOv8n model...")
    # yolov8n.pt is the nano model, lightest and fastest for training
    model = YOLO("yolov8n.pt")
    
    print(f"\n--- Starting Custom Face Detection Training ---")
    print(f"Dataset configuration: {yaml_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch}")
    print(f"Device: CPU (Forced for compatibility)")
    print("------------------------------------------------\n")
    
    # Train the model
    try:
        model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=batch,
            device='cpu',  # Force CPU
            workers=0,     # Disable multiprocessing workers to prevent Windows-specific PyTorch errors
            project='runs',
            name='face_detection_train'
        )
        print("\nTraining completed successfully!")
        print("Your trained model weights are saved at:")
        print(f"  {os.path.abspath('runs/face_detection_train/weights/best.pt')}")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom YOLOv8 face detector on the mini demo dataset.")
    parser.add_argument('--epochs', type=int, default=1, 
                        help="Number of epochs to train for. Set to 1-2 for a quick demo on CPU. (default: 1)")
    parser.add_argument('--batch', type=int, default=8,
                        help="Batch size for training. (default: 8)")
    
    args = parser.parse_args()
    train_model(args.epochs, args.batch)
