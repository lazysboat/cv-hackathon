import os
import sys
import yaml
import argparse
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
    
    # Create dataset config with minimal parameters
    dataset_config = {
        'path': os.path.join(script_dir, 'train'),  # Path to the root directory of the dataset
        'train': {
            'path': os.path.join('train', 'images'),  # Path to train images
            'masks_path': os.path.join('train', 'masks')  # Path to train masks
        },
        'val': {
            'path': os.path.join('train', 'images'),  # Same as train for now
            'masks_path': os.path.join('train', 'masks')  # Same as train for now
        },
        'names': {0: 'object'}  # Default class name
    }
    
    # Write the configuration to a YAML file in the script directory
    yaml_path = os.path.join(script_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset YAML configuration created successfully at {yaml_path}")
    return yaml_path

def train_minimal(data_yaml=None, masks_dir=None):
    """
    Train YOLO segmentation model with minimal parameters
    
    Args:
        data_yaml: Path to a custom YAML config file (optional)
        masks_dir: Path to a custom masks directory (optional)
    """
    # Setup dataset configuration if not provided
    try:
        if data_yaml is None:
            dataset_yaml = setup_dataset_yaml()
        else:
            if os.path.exists(data_yaml):
                dataset_yaml = data_yaml
                print(f"Using provided YAML config: {data_yaml}")
            else:
                print(f"Warning: Provided YAML config {data_yaml} not found!")
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
        print("Please download the YOLOv11 or YOLOv8 segmentation weights file")
        print("and place it in the Yolov11 directory.")
        return
    
    # Load YOLO model
    try:
        model = YOLO(model_file)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Configure minimal training parameters
    print("Starting YOLO segmentation training with minimal parameters...")
    try:
        # Use only essential parameters to avoid compatibility issues
        model.train(
            data=dataset_yaml,          # Path to data config file
            epochs=5,                   # Number of epochs to train for
            imgsz=640,                  # Image size
            batch=8,                    # Batch size (reduced for CPU training)
            device='cpu',               # Use CPU instead of GPU
            name='yolo_seg_minimal',    # Name of the training experiment
            project='runs/segment',     # Project name
            exist_ok=True,              # Overwrite the output directory
            verbose=True,               # Verbose output to help debug
            single_cls=True,            # Treat as single class problem
            rect=False,                 # Don't use rectangular training
            cache=False                 # Disable caching which might cause issues
        )
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        print("\nTrying with even fewer parameters...")
        try:
            # Fallback to absolute minimum parameters
            model.train(
                data=dataset_yaml,
                epochs=5,
                device='cpu',
                imgsz=640,
                batch=4,        # Very small batch size
                single_cls=True  # Single class for simpler training
            )
            print("Training completed successfully!")
        except Exception as e:
            print(f"Error during minimal training: {e}")
            print("Please check the ultralytics documentation for the latest API.")
            print("You may need to check your dataset format: Try running check_dataset.py")

def main():
    """
    Main function to execute the minimal training process
    """
    parser = argparse.ArgumentParser(description="Train YOLO segmentation with minimal parameters")
    parser.add_argument("--data", type=str, help="Path to dataset.yaml file")
    parser.add_argument("--masks", type=str, help="Path to custom masks directory")
    args = parser.parse_args()
    
    print("=== YOLO Segmentation Minimal Training ===")
    print(f"Current working directory: {os.getcwd()}")
    
    train_minimal(data_yaml=args.data, masks_dir=args.masks)

if __name__ == "__main__":
    main() 