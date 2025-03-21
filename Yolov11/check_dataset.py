import os
import sys
import glob
import numpy as np
from PIL import Image
import yaml

def check_dataset_structure():
    """
    Check if the dataset structure is valid for YOLO segmentation
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define expected directories
    train_images_dir = os.path.join(script_dir, 'train', 'train', 'images')
    train_masks_dir = os.path.join(script_dir, 'train', 'train', 'masks')
    
    # Check if directories exist
    if not os.path.exists(train_images_dir):
        print(f"ERROR: Images directory {train_images_dir} does not exist!")
        return False
    
    if not os.path.exists(train_masks_dir):
        print(f"ERROR: Masks directory {train_masks_dir} does not exist!")
        return False
    
    # List images and masks
    image_files = sorted(glob.glob(os.path.join(train_images_dir, '*.*')))
    mask_files = sorted(glob.glob(os.path.join(train_masks_dir, '*.*')))
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    if len(image_files) == 0:
        print("ERROR: No image files found!")
        return False
    
    if len(mask_files) == 0:
        print("ERROR: No mask files found!")
        return False
    
    # Check image-mask pairs
    image_basenames = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    mask_basenames = [os.path.splitext(os.path.basename(f))[0] for f in mask_files]
    
    # Find missing masks
    missing_masks = set(image_basenames) - set(mask_basenames)
    if missing_masks:
        print(f"WARNING: {len(missing_masks)} images don't have matching masks")
        for name in list(missing_masks)[:5]:  # Show first 5
            print(f"  - Missing mask for: {name}")
        if len(missing_masks) > 5:
            print(f"  - ... and {len(missing_masks) - 5} more")
    
    # Find extra masks
    extra_masks = set(mask_basenames) - set(image_basenames)
    if extra_masks:
        print(f"WARNING: {len(extra_masks)} masks don't have matching images")
        for name in list(extra_masks)[:5]:  # Show first 5
            print(f"  - Extra mask: {name}")
        if len(extra_masks) > 5:
            print(f"  - ... and {len(extra_masks) - 5} more")
    
    # Check image formats
    valid_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    invalid_images = [f for f in image_files if os.path.splitext(f)[1].lower() not in valid_image_extensions]
    if invalid_images:
        print(f"WARNING: {len(invalid_images)} images have unsupported formats")
        for img in invalid_images[:5]:
            print(f"  - Invalid image format: {os.path.basename(img)}")
    
    # Check mask formats and content
    print("\nChecking mask content...")
    mask_issues = []
    valid_pairs = []
    
    # Find matching pairs to check
    matched_pairs = []
    for img_file in image_files:
        img_name = os.path.splitext(os.path.basename(img_file))[0]
        
        # Try to find corresponding mask with any extension
        mask_candidates = [f for f in mask_files if os.path.splitext(os.path.basename(f))[0] == img_name]
        if mask_candidates:
            matched_pairs.append((img_file, mask_candidates[0]))
    
    # Only check first 10 masks to avoid long processing
    for i, (img_file, mask_file) in enumerate(matched_pairs[:10]):
        try:
            img = Image.open(img_file)
            mask = Image.open(mask_file)
            
            # Check dimensions
            if img.size != mask.size:
                mask_issues.append(f"{os.path.basename(mask_file)}: Dimensions mismatch - Image: {img.size}, Mask: {mask.size}")
                continue
            
            # Check if mask is binary
            mask_array = np.array(mask)
            unique_values = np.unique(mask_array)
            
            if not ((len(unique_values) <= 2) and (0 in unique_values) and (1 in unique_values or 255 in unique_values)):
                mask_issues.append(f"{os.path.basename(mask_file)}: Not binary - Contains values {unique_values}")
                continue
            
            valid_pairs.append((img_file, mask_file))
            
        except Exception as e:
            mask_issues.append(f"{os.path.basename(mask_file)}: Error - {str(e)}")
    
    if mask_issues:
        print(f"WARNING: Found {len(mask_issues)} problematic masks:")
        for issue in mask_issues:
            print(f"  - {issue}")
    
    if len(valid_pairs) == 0:
        print("ERROR: No valid image-mask pairs found!")
        return False
    
    print(f"\nFound at least {len(valid_pairs)} valid image-mask pairs for training")
    
    return True

def create_fixed_yaml():
    """
    Create a more compatible YAML file for YOLOv8 segmentation
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a train-val split for better compatibility
    # YOLOv8 might expect both train and val sets
    dataset_config = {
        'path': os.path.join(script_dir, 'train'),  # Root path
        'train': {
            'path': os.path.join('train', 'images'),
            'masks_path': os.path.join('train', 'masks')
        },
        'val': {
            'path': os.path.join('train', 'images'),  # Same as train for now
            'masks_path': os.path.join('train', 'masks')
        },
        'names': {0: 'object'}
    }
    
    yaml_path = os.path.join(script_dir, 'dataset_fixed.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created fixed YAML configuration at {yaml_path}")
    return yaml_path

def fix_mask_format():
    """
    Convert masks to the right format if needed
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    masks_dir = os.path.join(script_dir, 'train', 'train', 'masks')
    fixed_dir = os.path.join(script_dir, 'train', 'train', 'masks_fixed')
    
    if not os.path.exists(masks_dir):
        print("Masks directory not found")
        return False
    
    # Create the output directory if it doesn't exist
    os.makedirs(fixed_dir, exist_ok=True)
    
    mask_files = glob.glob(os.path.join(masks_dir, '*.*'))
    print(f"Found {len(mask_files)} masks to process")
    
    converted = 0
    for mask_file in mask_files:
        try:
            mask = np.array(Image.open(mask_file))
            mask_name = os.path.basename(mask_file)
            output_path = os.path.join(fixed_dir, mask_name)
            
            # Ensure mask is binary (0 and 255)
            unique_values = np.unique(mask)
            
            if 1 in unique_values and 255 not in unique_values:
                # Convert 0/1 mask to 0/255
                mask = mask * 255
                converted += 1
            
            # Ensure it's saved as grayscale
            Image.fromarray(mask.astype(np.uint8), mode='L').save(output_path)
            
        except Exception as e:
            print(f"Error processing {os.path.basename(mask_file)}: {e}")
    
    print(f"Converted {converted} masks to 0/255 format")
    print(f"All masks processed and saved to {fixed_dir}")
    
    # If we fixed any masks, update the original masks_path
    if converted > 0:
        print("Some masks were converted - consider using the fixed masks for training")
        return fixed_dir
    
    return None

def main():
    print("=== Checking YOLO Segmentation Dataset ===")
    
    if not check_dataset_structure():
        print("\nDataset structure has issues. Please fix them before training.")
        return
    
    print("\nChecking mask formats...")
    fixed_masks_dir = fix_mask_format()
    
    print("\nCreating compatible YAML configuration...")
    yaml_path = create_fixed_yaml()
    
    print("\n=== Summary ===")
    print("Dataset structure seems valid for YOLO segmentation")
    if fixed_masks_dir:
        print(f"Fixed masks available at: {fixed_masks_dir}")
    print(f"Compatible YAML configuration created at: {yaml_path}")
    print("\nSuggested training command:")
    print("python train_yolov11_minimal.py --data dataset_fixed.yaml")

if __name__ == "__main__":
    main() 