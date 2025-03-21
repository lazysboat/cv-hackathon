import os
import numpy as np
from PIL import Image
import glob

def fix_specific_masks():
    """Fix specific masks identified as problematic"""
    # Create output directory for fixed masks
    script_dir = os.path.dirname(os.path.abspath(__file__))
    masks_dir = os.path.join(script_dir, 'train', 'train', 'masks')
    fixed_dir = os.path.join(script_dir, 'train', 'train', 'masks_fixed')
    os.makedirs(fixed_dir, exist_ok=True)
    
    # Lists of masks to fix
    # RGB masks to convert to grayscale binary
    rgb_masks = [
        r"image_0.png",
        r"image_10.png",
        r"image_10001.png",
        r"image_10002.png",
        r"image_10003.png",
        r"image_10004.png",
        r"image_10006.png",
        r"image_10007.png",
        r"image_10008.png",
        r"image_10009.png",
        r"image_1001.png",
        r"image_10010.png",
        r"image_10011.png",
        r"image_10012.png",
        r"image_10013.png",
    ]

    nonbinary_masks = []

    # Problem masks to fix
    problem_masks = [
        r"image_0.png",
        r"image_10.png",
        r"image_10001.png",
        r"image_10002.png",
        r"image_10003.png",
        r"image_10004.png",
        r"image_10006.png",
        r"image_10007.png",
        r"image_10008.png",
        r"image_10009.png",
        r"image_1001.png",
        r"image_10010.png",
        r"image_10011.png",
        r"image_10012.png",
        r"image_10013.png",
    ]

    # Process RGB masks
    for mask_name in rgb_masks:
        try:
            mask_path = os.path.join(masks_dir, mask_name)
            output_path = os.path.join(fixed_dir, mask_name)
            
            if not os.path.exists(mask_path):
                print(f"Warning: {mask_path} not found")
                continue
            
            print(f"Converting RGB mask: {mask_name}")
            # Open the image
            img = Image.open(mask_path)
            # Convert to grayscale
            gray = img.convert('L')
            # Convert to binary (0/255)
            binary_array = np.array(gray)
            binary_array = (binary_array > 127).astype(np.uint8) * 255
            # Save
            Image.fromarray(binary_array, mode='L').save(output_path)
            print(f"  Saved to {output_path}")
        except Exception as e:
            print(f"Error fixing {mask_name}: {e}")
    
    # Process grayscale non-binary masks
    for mask_name in nonbinary_masks:
        try:
            mask_path = os.path.join(masks_dir, mask_name)
            output_path = os.path.join(fixed_dir, mask_name)
            
            if not os.path.exists(mask_path):
                print(f"Warning: {mask_path} not found")
                continue
            
            print(f"Converting non-binary mask: {mask_name}")
            # Open the image
            img = Image.open(mask_path)
            # Convert to binary (0/255)
            binary_array = np.array(img)
            binary_array = (binary_array > 127).astype(np.uint8) * 255
            # Save
            Image.fromarray(binary_array, mode='L').save(output_path)
            print(f"  Saved to {output_path}")
        except Exception as e:
            print(f"Error fixing {mask_name}: {e}")
    
    # Process problem masks
    for mask_name in problem_masks:
        try:
            mask_path = os.path.join(masks_dir, mask_name)
            output_path = os.path.join(fixed_dir, mask_name)
            
            if not os.path.exists(mask_path):
                print(f"Warning: {mask_path} not found")
                continue
            
            print(f"Fixing problem mask: {mask_name}")
            # Try to convert in the most general way
            img = Image.open(mask_path)
            # First convert to grayscale if it's not
            if img.mode != 'L':
                img = img.convert('L')
            # Then make it binary
            binary_array = np.array(img)
            binary_array = (binary_array > 127).astype(np.uint8) * 255
            # Save
            Image.fromarray(binary_array, mode='L').save(output_path)
            print(f"  Saved to {output_path}")
        except Exception as e:
            print(f"Error fixing {mask_name}: {e}")
    
    # Count fixed masks
    fixed_count = len(glob.glob(os.path.join(fixed_dir, '*.*')))
    print(f"
Fixed {fixed_count} masks. All saved to {fixed_dir}")
    
    # Ask if user wants to replace originals
    print("
Do you want to replace the original masks with the fixed ones? (y/n)")
    response = input("> ")
    
    if response.lower() == 'y':
        # Backup originals first
        backup_dir = os.path.join(script_dir, 'train', 'train', 'masks_backup')
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy originals to backup
        for mask_path in glob.glob(os.path.join(masks_dir, '*.*')):
            mask_name = os.path.basename(mask_path)
            backup_path = os.path.join(backup_dir, mask_name)
            try:
                import shutil
                shutil.copy2(mask_path, backup_path)
            except Exception as e:
                print(f"Error backing up {mask_name}: {e}")
        
        print(f"Original masks backed up to {backup_dir}")
        
        # Replace with fixed masks
        for fixed_path in glob.glob(os.path.join(fixed_dir, '*.*')):
            fixed_name = os.path.basename(fixed_path)
            original_path = os.path.join(masks_dir, fixed_name)
            try:
                import shutil
                shutil.copy2(fixed_path, original_path)
            except Exception as e:
                print(f"Error replacing {fixed_name}: {e}")
        
        print("Original masks replaced with fixed ones")
    else:
        print("
Fixed masks are available at:")
        print(fixed_dir)
        print("You can manually copy them to replace the originals if needed")

if __name__ == "__main__":
    print("=== Fixing Specific Masks ===")
    fix_specific_masks()
    print("
After fixing masks, try training with:")
    print("python train_simple.py --batch 1 --epochs 1")
