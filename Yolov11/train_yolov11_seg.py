import os
import yaml
from ultralytics import YOLO

def setup_dataset_yaml():
    """
    Create YAML configuration file for the dataset
    """
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define dataset paths - adjust based on relative structure
    train_images_dir = os.path.join(script_dir, 'train', 'train', 'images')
    train_masks_dir = os.path.join(script_dir, 'train', 'train', 'masks')
    
    # Check if directories exist
    if not os.path.exists(train_images_dir):
        print(f"Warning: {train_images_dir} does not exist!")
        print("Please ensure your dataset is structured as Yolov11/train/train/images/")
        raise FileNotFoundError(f"Images directory not found at {train_images_dir}")
    
    if not os.path.exists(train_masks_dir):
        print(f"Warning: {train_masks_dir} does not exist!")
        print("Please ensure your dataset is structured as Yolov11/train/train/masks/")
        raise FileNotFoundError(f"Masks directory not found at {train_masks_dir}")
    
    # Count files to confirm dataset exists
    image_files = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    mask_files = [f for f in os.listdir(train_masks_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    if len(image_files) == 0:
        print("Warning: No image files found in the images directory!")
    if len(mask_files) == 0:
        print("Warning: No mask files found in the masks directory!")
    
    # Create dataset config - use absolute paths to avoid confusion
    dataset_config = {
        'path': os.path.join(script_dir, 'train'),  # Path to the root directory of the dataset
        'train': os.path.join('train', 'images'),  # Path to train images (relative to 'path')
        'val': os.path.join('train', 'images'),  # Using same for validation as we don't have a separate val folder
        'test': None,  # No test split for now
        'names': {0: 'object'},  # Default class name, update with your classes
        'masks_path': os.path.join('train', 'masks')  # Path to the binary masks folder
    }
    
    # Write the configuration to a YAML file in the script directory
    yaml_path = os.path.join(script_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset YAML configuration created successfully at {yaml_path}")
    return yaml_path

def train_yolov11_seg():
    """
    Train YOLOv11-seg model on the dataset
    """
    # Setup dataset configuration
    try:
        dataset_yaml = setup_dataset_yaml()
    except Exception as e:
        print(f"Error setting up dataset: {e}")
        return
    
    # Check for model weights file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_paths = [
        os.path.join(script_dir, "yolo11n-seg.pt"),          # Look in script directory
        os.path.join(script_dir, "yolov8n-seg.pt"),          # Try YOLOv8 as fallback
        os.path.join(os.path.dirname(script_dir), "yolo11n-seg.pt"),  # Look in parent directory
        "yolo11n-seg.pt"  # Look in current working directory
    ]
    
    model_file = None
    for path in model_paths:
        if os.path.exists(path):
            model_file = path
            print(f"Found YOLO model at: {path}")
            break
    
    if model_file is None:
        print("Error: YOLO model weights not found!")
        print("Please download the YOLOv11 weights file (yolo11n-seg.pt) and place it in the Yolov11 directory.")
        print("You can obtain this file from the official YOLO repository.")
        return
    
    # Load YOLOv11-seg model
    try:
        model = YOLO(model_file)  # Load the YOLO segment model
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Make sure you have the YOLO weights file at {model_file}")
        return
    
    # Configure training parameters - updated with only compatible parameters
    print("Starting YOLO segmentation training...")
    try:
        # Use a reduced set of parameters that are known to be compatible
        model.train(
            data=dataset_yaml,          # Path to data config file
            epochs=5,                   # Number of epochs to train for (reduced from 100 to 5)
            imgsz=640,                  # Image size
            batch=16,                   # Batch size
            device='0',                 # GPU device, use '' for CPU
            workers=8,                  # Number of dataloader workers
            name='yolov11_seg_train',   # Name of the training experiment
            project='runs/segment',     # Project name
            exist_ok=True,              # Overwrite the output directory
            pretrained=True,            # Use pretrained weights
            optimizer='auto',           # Optimizer selection
            verbose=True,               # Verbose output
            seed=0,                     # Random seed
            patience=100,               # Early stopping patience (epochs without improvement)
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        print("Double-check your dataset structure and model configurations.")
        print("\nIf you're still seeing parameter errors, please try running the minimal version:")
        print("python train_yolov11_minimal.py")

def main():
    """
    Main function to execute the training process
    """
    print("=== YOLOv11 Segmentation Training ===")
    print(f"Current working directory: {os.getcwd()}")
    train_yolov11_seg()

if __name__ == "__main__":
    main() 