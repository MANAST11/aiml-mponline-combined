# Face Detection

A comprehensive Python suite for running real-time face detection directly inside your local IDE using two distinct approaches:
1. **Mediapipe Face Detector (Instant - Recommended):** Real-time, highly accurate face detection using Google's pre-trained model. Runs instantly on CPU with zero training.
2. **YOLOv8 Custom Face Detector:** A workflow utilizing the Ultralytics framework, configured to let you prepare a small subset of the Kaggle dataset, run a fast CPU training demo, and perform inference using your custom weights.

Dataset source: [Kaggle Face Detection Dataset by freak2209](https://www.kaggle.com/datasets/freak2209/face-data).

---

## Features

- **Webcam Mode:** Run live face detection using your camera feed.
- **Kaggle Dataset mode:** Cycles through images from the Kaggle dataset automatically.
- **Custom Inputs:** Test on local image or video files.
- **Premium Visualization:** Sleek, high-contrast bounding boxes with corner accents and confidence level indicators.
- **CPU Optimized:** Training configurations (batch size, workers, device) are tailored to run smoothly on local environments without requiring a GPU.

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/MANAST11/face-detection
   cd <repository-folder>
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.8+ installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Method A: Mediapipe Face Detection (Instant, No Training Required)

Run the pre-trained Mediapipe model on different sources:

* **Cycle through Kaggle dataset images:**
  ```bash
  python detect_mediapipe.py --source dataset
  ```
  *Press **any key** (except Q/ESC) to view the next image. Press **Q** or **ESC** to exit.*

* **Detect faces on live Webcam feed:**
  ```bash
  python detect_mediapipe.py --source webcam
  ```
  *Press **Q** or **ESC** to exit.*

* **Detect faces on a local image/video file:**
  ```bash
  python detect_mediapipe.py --source path/to/file.jpg
  ```

---

### Method B: Custom YOLOv8 Training & Detection (Demo Setup)

Follow these steps to prepare a mini-dataset, train a custom model, and run detection:

1. **Prepare a Mini Dataset:**
   Extracts a small subset from the Kaggle dataset (e.g. 40 training and 10 validation images) and configures paths:
   ```bash
   python prepare_demo_dataset.py --train 40 --val 10
   ```

2. **Train the Custom YOLOv8 Model:**
   Starts a training session on CPU for a demo epoch (you can increase the `--epochs` argument as needed):
   ```bash
   python train_yolo.py --epochs 1 --batch 8
   ```
   *Your custom weights will be saved to `runs/face_detection_train/weights/best.pt`.*

3. **Run Detection using Custom Weights:**
   * **On Kaggle Dataset:**
     ```bash
     python detect_yolo.py --source dataset
     ```
   * **On Webcam:**
     ```bash
     python detect_yolo.py --source webcam
     ```
   * **Using custom weights path or confidence threshold:**
     ```bash
     python detect_yolo.py --source webcam --weights runs/face_detection_train/weights/best.pt --confidence 0.3
     ```

---

## File Structure

```
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── detect_mediapipe.py         # Mediapipe face detector script
├── prepare_demo_dataset.py     # Script to extract and prepare mini dataset
├── train_yolo.py               # Custom YOLOv8 training script
└── detect_yolo.py              # YOLOv8 face detection inference script
```

## Controls

When the GUI window is open:
- Press **`q`** or **`ESC`** to close the program window.
- In **`dataset`** mode, press **any other key** to advance to the next image.
