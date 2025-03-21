import os
import glob
import numpy as np
from PIL import Image
import shutil

def fix_all_masks():
    """
    Convert all masks to grayscale binary format (0/255) without removing any data.
    This preserves the original masks by making backups, then converts all masks
    to the correct format for YOLO segmentation.
    """
    print("=== Fixing Mask Formats for YOLO Segmentation ===")
    
    # Get directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    masks_dir = os.path.join(script_dir, 'train', 'train', 'masks')
    
    # Verify that masks directory exists
    if not os.path.exists(masks_dir):
        print(f"ERROR: Masks directory not found at {masks_dir}")
        print("Make sure your dataset is structured as Yolov11/train/train/masks/")
        return
    
    # Create backup directory
    backup_dir = os.path.join(script_dir, 'train', 'train', 'masks_backup')
    os.makedirs(backup_dir, exist_ok=True)
    
    # Get list of mask files
    mask_files = glob.glob(os.path.join(masks_dir, '*.*'))
    if not mask_files:
        print(f"No mask files found in {masks_dir}")
        return
    
    print(f"Found {len(mask_files)} mask files to process")
    
    # First backup all original masks
    print(f"\nStep 1: Backing up original masks to {backup_dir}")
    for i, mask_file in enumerate(mask_files):
        mask_name = os.path.basename(mask_file)
        backup_path = os.path.join(backup_dir, mask_name)
        
        try:
            shutil.copy2(mask_file, backup_path)
            print(f"  Backed up {i+1}/{len(mask_files)}: {mask_name}", end='\r')
        except Exception as e:
            print(f"\nError backing up {mask_name}: {e}")
    
    print(f"\nAll {len(mask_files)} masks backed up successfully")
    
    # Process and convert each mask
    print("\nStep 2: Converting masks to grayscale binary format")
    converted_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, mask_file in enumerate(mask_files):
        try:
            mask_name = os.path.basename(mask_file)
            print(f"  Processing {i+1}/{len(mask_files)}: {mask_name}", end='\r')
            
            # Open the mask
            mask = Image.open(mask_file)
            
            # Get mask info
            mask_array = np.array(mask)
            dimensions = len(mask_array.shape)
            
            # Check if conversion is needed
            needs_conversion = False
            
            # Case 1: It's an RGB or RGBA image (3 dimensions)
            if dimensions == 3:
                needs_conversion = True
            # Case 2: It's grayscale but not binary
            elif dimensions == 2:
                unique_values = np.unique(mask_array)
                is_binary = len(unique_values) <= 2 and set(unique_values).issubset({0, 1, 255})
                needs_conversion = not is_binary
            # Case 3: Unusual format
            else:
                needs_conversion = True
            
            if needs_conversion:
                # Convert to grayscale if needed
                if mask.mode != 'L':
                    gray_mask = mask.convert('L')
                else:
                    gray_mask = mask
                
                # Convert to binary (0/255)
                gray_array = np.array(gray_mask)
                binary_array = (gray_array > 127).astype(np.uint8) * 255
                
                # Save back to the original location
                Image.fromarray(binary_array, mode='L').save(mask_file)
                converted_count += 1
            else:
                skipped_count += 1
                
        except Exception as e:
            print(f"\nError processing {os.path.basename(mask_file)}: {e}")
            error_count += 1
    
    print("\n" + "=" * 50)
    print(f"Conversion complete. Summary:")
    print(f"  - Total masks: {len(mask_files)}")
    print(f"  - Converted to binary format: {converted_count}")
    print(f"  - Already in correct format: {skipped_count}")
    print(f"  - Errors: {error_count}")
    print(f"  - Original backups: {backup_dir}")
    print("=" * 50)
    
    # Create a simple YAML configuration
    create_simple_yaml(script_dir)
    
    print("\nNext steps:")
    print("1. Try training with the simplified script:")
    print("   python train_simple.py --batch 1 --epochs 1")
    print("\nIf you still encounter issues, you can restore the original masks from the backup directory.")

def create_simple_yaml(script_dir):
    """Create a simple YAML config file for training with the fixed masks"""
    import yaml
    
    train_dir = os.path.join(script_dir, 'train', 'train')
    yaml_path = os.path.join(script_dir, 'simple_dataset.yaml')
    
    # Count classes from dataset
    classes = 1  # Default to 1 class if we can't determine
    
    # Create simple YAML config
    config = {
        'path': os.path.join(script_dir, 'train'),
        'train': os.path.join('train', 'images'),
        'val': os.path.join('train', 'images'),  # Using same for validation
        'test': None,
        'names': {0: 'object'},  # Default class name
        'nc': classes
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nCreated simple YAML config: {yaml_path}")

if __name__ == "__main__":
    fix_all_masks() 