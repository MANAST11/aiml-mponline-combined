import os
import shutil
import glob
import kagglehub
import argparse

def prepare_demo(num_train=40, num_val=10):
    print("Locating Kaggle dataset...")
    try:
        dataset_path = kagglehub.dataset_download('freak2209/face-data')
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return
        
    src_images_dir = os.path.join(dataset_path, 'Custom_Data', 'images', 'train')
    src_labels_dir = os.path.join(dataset_path, 'Custom_Data', 'labels', 'train')
    
    if not os.path.exists(src_images_dir) or not os.path.exists(src_labels_dir):
        print("Error: Could not find Kaggle dataset directories.")
        return
        
    # Get all source images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    src_images = []
    for ext in image_extensions:
        src_images.extend(glob.glob(os.path.join(src_images_dir, ext)))
    src_images.sort()
    
    total_needed = num_train + num_val
    if len(src_images) < total_needed:
        print(f"Warning: Dataset only has {len(src_images)} images, requesting {total_needed}.")
        num_train = int(len(src_images) * 0.8)
        num_val = len(src_images) - num_train
        total_needed = num_train + num_val
        
    print(f"Preparing a mini demo dataset with {num_train} train images and {num_val} val images...")
    
    # Destination directories
    base_dir = os.path.abspath('demo_dataset')
    dest_dirs = {
        'train_img': os.path.join(base_dir, 'images', 'train'),
        'val_img': os.path.join(base_dir, 'images', 'val'),
        'train_lbl': os.path.join(base_dir, 'labels', 'train'),
        'val_lbl': os.path.join(base_dir, 'labels', 'val')
    }
    
    # Create directories
    for path in dest_dirs.values():
        os.makedirs(path, exist_ok=True)
        # Clear existing files if any
        for f in glob.glob(os.path.join(path, '*')):
            if os.path.isfile(f):
                os.remove(f)
                
    # Copy train set
    copied_train = 0
    for i in range(num_train):
        img_path = src_images[i]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(src_labels_dir, base_name + '.txt')
        
        # Check if label exists
        if os.path.exists(lbl_path):
            shutil.copy(img_path, os.path.join(dest_dirs['train_img'], os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(dest_dirs['train_lbl'], base_name + '.txt'))
            copied_train += 1
            
    # Copy val set
    copied_val = 0
    for i in range(num_train, num_train + num_val):
        if i >= len(src_images):
            break
        img_path = src_images[i]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(src_labels_dir, base_name + '.txt')
        
        # Check if label exists
        if os.path.exists(lbl_path):
            shutil.copy(img_path, os.path.join(dest_dirs['val_img'], os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(dest_dirs['val_lbl'], base_name + '.txt'))
            copied_val += 1
            
    print(f"Copied {copied_train} train images and {copied_val} val images.")
    
    # Write data.yaml file
    base_dir_formatted = base_dir.replace('\\', '/')
    yaml_content = f"""path: {base_dir_formatted}  # dataset root dir
train: images/train  # train images (relative to path)
val: images/val  # val images (relative to path)

# Classes
names:
  0: face
"""
    yaml_path = os.path.join(base_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"Created YOLOv8 dataset configuration file at: {yaml_path}")
    print("\nDataset preparation complete! You can now run 'python train_yolo.py' to start a fast demo training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a tiny face-data subset for training demonstration.")
    parser.add_argument('--train', type=int, default=40, help="Number of training images (default: 40)")
    parser.add_argument('--val', type=int, default=10, help="Number of validation images (default: 10)")
    
    args = parser.parse_args()
    prepare_demo(args.train, args.val)
