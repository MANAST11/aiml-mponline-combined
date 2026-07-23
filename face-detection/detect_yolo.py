import os
import sys
import glob
import cv2
from ultralytics import YOLO
import argparse
import kagglehub

def get_dataset_images():
    """Resolves the Kaggle dataset path and returns a list of image file paths."""
    print("Locating Kaggle dataset...")
    try:
        dataset_path = kagglehub.dataset_download('freak2209/face-data')
        images_dir = os.path.join(dataset_path, 'Custom_Data', 'images', 'train')
        if not os.path.exists(images_dir):
            print(f"Error: Images directory not found at {images_dir}")
            return []
        
        # Support common image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(images_dir, ext)))
        
        images.sort()
        return images
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return []

def draw_beautiful_box(img, x1, y1, x2, y2, label=None, color=(0, 255, 0), thickness=2):
    """Draws a premium-looking bounding box with corner accents and label."""
    # Main rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    # Draw thicker corner accents for a sleek aesthetic
    line_len = min(x2 - x1, y2 - y1) // 5
    accent_thick = thickness + 1
    
    # Top-left corner
    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, accent_thick)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, accent_thick)
    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, accent_thick)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, accent_thick)
    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, accent_thick)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, accent_thick)
    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, accent_thick)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, accent_thick)
    
    # Label drawing
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
        
        # Make a filled background block for label text
        cv2.rectangle(img, (x1, y1 - text_height - 6), (x1 + text_width + 6, y1), color, -1)
        
        # Put text (contrast color: dark blue/black for bright background)
        text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        cv2.putText(img, label, (x1 + 3, y1 - 4), font, font_scale, text_color, font_thick, cv2.LINE_AA)

def run_yolo_detection(weights_path, source, min_confidence):
    # Check if weights exist
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at: {weights_path}")
        print("If you haven't trained a custom model, run 'python train_yolo.py' first.")
        print("Alternatively, you can test with pre-trained yolov8n.pt by passing '--weights yolov8n.pt'")
        return
        
    print(f"Loading custom YOLOv8 model from: {weights_path}...")
    model = YOLO(weights_path)
    
    print("\n--- Controls ---")
    print("Press 'q' or 'ESC' to exit the program.")
    if source == 'dataset':
        print("Press ANY OTHER KEY to view the next image.")
    print("----------------\n")
    
    if source == 'dataset':
        image_paths = get_dataset_images()
        if not image_paths:
            print("No images found in the dataset.")
            return
            
        print(f"Loaded {len(image_paths)} images from dataset.")
        
        for idx, img_path in enumerate(image_paths):
            print(f"[{idx+1}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w, c = img.shape
            
            # Run YOLOv8 inference
            results = model(img, conf=min_confidence, verbose=False)
            
            # Draw detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    score = box.conf[0].item()
                    label = f"Face: {score:.2f}"
                    draw_beautiful_box(img, x1, y1, x2, y2, label=label)
            
            # Show image
            window_name = "Kaggle Dataset Custom YOLO Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, img)
            
            # Wait for key press
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q') or key & 0xFF == 27:
                break
                
        cv2.destroyAllWindows()
        
    else:
        # Webcam or Video source
        is_webcam = False
        if source == 'webcam' or source == '0':
            video_source = 0
            is_webcam = True
            print("Opening Webcam...")
        else:
            video_source = source
            print(f"Opening Video File: {source}")
            
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open source: {source}")
            return
            
        window_name = "Live Face Detection (Custom YOLO)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    print("Error: Failed to grab frame from webcam.")
                else:
                    print("End of video file reached.")
                break
                
            # If webcam, flip horizontally for natural mirror feel
            if is_webcam:
                frame = cv2.flip(frame, 1)
                
            # Run YOLOv8 inference
            results = model(frame, conf=min_confidence, verbose=False)
            
            # Draw detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    score = box.conf[0].item()
                    label = f"Face: {score:.2f}"
                    draw_beautiful_box(frame, x1, y1, x2, y2, label=label)
            
            # Add status text
            status_text = "Live Stream (YOLO)" if is_webcam else "Video Stream (YOLO)"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow(window_name, frame)
            
            # Refresh every 1ms, check if 'q' is pressed
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference using Custom-Trained YOLOv8 Face Detection Model")
    parser.add_argument('--weights', type=str, default='runs/face_detection_train/weights/best.pt',
                        help="Path to the trained YOLO weights (.pt file). (default: runs/face_detection_train/weights/best.pt)")
    parser.add_argument('--source', type=str, default='dataset',
                        help="Source of inputs: 'dataset' (Kaggle), 'webcam' or '0' (live webcam), or path to an image/video file.")
    parser.add_argument('--confidence', type=float, default=0.25,
                        help="Minimum detection confidence threshold (default: 0.25)")
    
    args = parser.parse_args()
    run_yolo_detection(args.weights, args.source, args.confidence)
