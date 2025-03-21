import os
import glob
import numpy as np
from PIL import Image
import shutil

def fix_mask_dimensions():
    """
    Fix mask dimensions - convert RGB masks to grayscale binary masks
    and ensure they are in the correct format for YOLO segmentation
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    masks_dir = os.path.join(script_dir, 'train', 'train', 'masks')
    fixed_dir = os.path.join(script_dir, 'train', 'train', 'masks_fixed')
    
    if not os.path.exists(masks_dir):
        print(f"ERROR: Masks directory {masks_dir} not found!")
        return None
    
    # Create output directory
    os.makedirs(fixed_dir, exist_ok=True)
    
    # Get all mask files
    mask_files = glob.glob(os.path.join(masks_dir, '*.*'))
    print(f"Found {len(mask_files)} masks to process")
    
    fixed_count = 0
    
    for mask_file in mask_files:
        try:
            # Load the mask
            mask = Image.open(mask_file)
            mask_array = np.array(mask)
            mask_name = os.path.basename(mask_file)
            output_path = os.path.join(fixed_dir, mask_name)
            
            # Check dimensions
            if len(mask_array.shape) > 2:
                print(f"Converting {mask_name} from {len(mask_array.shape)}D to 2D")
                
                # Convert to grayscale if it's RGB
                if len(mask_array.shape) == 3:
                    # Convert RGB to grayscale
                    mask = mask.convert('L')
                    mask_array = np.array(mask)
                    fixed_count += 1
            
            # Ensure mask is binary (0 and 255)
            unique_values = np.unique(mask_array)
            print(f"Mask {mask_name} contains values: {unique_values}")
            
            # If mask has values other than 0 and 255, convert it
            if not ((len(unique_values) <= 2) and (0 in unique_values) and (255 in unique_values)):
                if 1 in unique_values:
                    # Convert 0/1 to 0/255
                    mask_array = (mask_array > 0).astype(np.uint8) * 255
                else:
                    # Threshold the grayscale image to create a binary mask
                    mask_array = (mask_array > 127).astype(np.uint8) * 255
                
                fixed_count += 1
            
            # Save the fixed mask in grayscale format
            Image.fromarray(mask_array.astype(np.uint8), mode='L').save(output_path)
            
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
    
    print(f"Fixed {fixed_count} masks. All masks saved to {fixed_dir}")
    
    if fixed_count > 0:
        # Create a backup of the original masks
        backup_dir = os.path.join(script_dir, 'train', 'train', 'masks_original')
        if not os.path.exists(backup_dir):
            print(f"Creating backup of original masks in {backup_dir}")
            os.makedirs(backup_dir, exist_ok=True)
            
            for mask_file in mask_files:
                mask_name = os.path.basename(mask_file)
                backup_path = os.path.join(backup_dir, mask_name)
                shutil.copy2(mask_file, backup_path)
            
            # Replace original masks with fixed ones
            print("Replacing original masks with fixed ones...")
            for fixed_mask in glob.glob(os.path.join(fixed_dir, '*.*')):
                mask_name = os.path.basename(fixed_mask)
                original_path = os.path.join(masks_dir, mask_name)
                shutil.copy2(fixed_mask, original_path)
            
            print("Original masks replaced with fixed versions.")
            print(f"Backup of original masks saved in {backup_dir}")
        
        return masks_dir
    
    return None

def create_minimal_yaml_file():
    """
    Create a simplified YAML file that works for YOLOv8 segmentation
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Very simple YAML config
    yaml_config = {
        # Define the paths as simple strings
        'train': os.path.join(script_dir, 'train', 'train', 'images'),
        'val': os.path.join(script_dir, 'train', 'train', 'images'),
        'names': {0: 'object'},
        'nc': 1
    }
    
    # Write to file
    yaml_path = os.path.join(script_dir, 'simple_dataset.yaml')
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    print(f"Created simplified YAML at {yaml_path}")
    return yaml_path

def main():
    print("=== Fixing Mask Dimensions for YOLO Segmentation ===")
    
    # Fix the mask dimensions
    fixed_dir = fix_mask_dimensions()
    
    # Create a minimal YAML file
    yaml_path = create_minimal_yaml_file()
    
    print("\n=== Instructions ===")
    print("Your masks have been fixed to the correct format for YOLO segmentation.")
    print("Try running the training with this command:")
    print(f"python train_yolov11_minimal.py --data {yaml_path}")
    print("\nIf you still encounter issues, try using an even simpler command:")
    print("python -c \"from ultralytics import YOLO; model = YOLO('yolov8n-seg.pt'); model.train(data='simple_dataset.yaml', epochs=1, imgsz=640, batch=1, device='cpu')\"")

if __name__ == "__main__":
    main() 