import os
import glob
import numpy as np
from PIL import Image

def inspect_masks():
    """
    Inspect masks to identify their format and dimensions
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    masks_dir = os.path.join(script_dir, 'train', 'train', 'masks')
    
    if not os.path.exists(masks_dir):
        print(f"ERROR: Masks directory {masks_dir} not found!")
        return
    
    # Get sample mask files (limit to 10 for brevity)
    mask_files = glob.glob(os.path.join(masks_dir, '*.*'))[:10]
    
    if not mask_files:
        print(f"No mask files found in {masks_dir}")
        return
    
    print(f"Examining {len(mask_files)} sample masks from {masks_dir}")
    print("-" * 50)
    
    # Track different types of masks
    rgb_masks = []
    grayscale_masks = []
    binary_masks = []
    other_masks = []
    
    for mask_file in mask_files:
        try:
            mask_name = os.path.basename(mask_file)
            print(f"\nAnalyzing {mask_name}:")
            
            # Load the mask with PIL
            mask = Image.open(mask_file)
            print(f"  Format: {mask.format}")
            print(f"  Mode: {mask.mode}")
            print(f"  Size: {mask.size}")
            
            # Convert to numpy array for deeper analysis
            mask_array = np.array(mask)
            
            # Check dimensions
            shape = mask_array.shape
            print(f"  Array shape: {shape}")
            
            if len(shape) == 2:
                print("  Dimensions: 2D (grayscale)")
                
                # Check unique values
                unique_values = np.unique(mask_array)
                print(f"  Unique values: {unique_values}")
                
                if len(unique_values) <= 2:
                    if set(unique_values).issubset({0, 255}) or set(unique_values).issubset({0, 1}):
                        print("  Type: Binary mask (good for segmentation)")
                        binary_masks.append(mask_file)
                    else:
                        print("  Type: Binary-like mask with non-standard values")
                        other_masks.append(mask_file)
                else:
                    print("  Type: Grayscale mask (non-binary)")
                    grayscale_masks.append(mask_file)
            
            elif len(shape) == 3:
                print("  Dimensions: 3D (likely RGB or RGBA)")
                
                # Check if it's a standard RGB/RGBA image
                if shape[2] == 3:
                    print("  Type: RGB mask (needs conversion to grayscale)")
                    rgb_masks.append(mask_file)
                elif shape[2] == 4:
                    print("  Type: RGBA mask (needs conversion to grayscale)")
                    rgb_masks.append(mask_file)
                else:
                    print(f"  Type: Unusual 3D array with {shape[2]} channels")
                    other_masks.append(mask_file)
            
            else:
                print(f"  Type: Unusual dimensions {len(shape)}D")
                other_masks.append(mask_file)
                
        except Exception as e:
            print(f"  ERROR processing {mask_name}: {e}")
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Binary masks: {len(binary_masks)}/{len(mask_files)}")
    print(f"RGB/RGBA masks: {len(rgb_masks)}/{len(mask_files)}")
    print(f"Grayscale (non-binary) masks: {len(grayscale_masks)}/{len(mask_files)}")
    print(f"Other mask types: {len(other_masks)}/{len(mask_files)}")
    
    print("\nRECOMMENDATIONS:")
    if rgb_masks:
        print("- You have RGB masks that need to be converted to grayscale binary format")
        print("  Run the fix_masks.py script to convert them")
    
    if grayscale_masks:
        print("- You have grayscale masks that need to be converted to binary (0/255) format")
        print("  Run the fix_masks.py script to convert them")
    
    if other_masks:
        print("- You have some unusual mask formats that might cause issues")
        print("  Examine these masks manually and consider regenerating them")
    
    if binary_masks and len(binary_masks) == len(mask_files):
        print("- All your masks appear to be in the correct binary format!")
        print("  If you're still having issues, there might be other problems with the dataset")

def main():
    print("=== YOLO Segmentation Mask Inspector ===")
    inspect_masks()
    
    print("\nIf your masks need fixing, run the fix_masks.py script:")
    print("python fix_masks.py")
    
    print("\nAfter fixing masks, try training with the simplified script:")
    print("python train_simple.py --epochs 1 --batch 1")

if __name__ == "__main__":
    main() 