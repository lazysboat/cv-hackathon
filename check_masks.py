import os
import numpy as np
from PIL import Image

def check_mask_has_ai(mask_path):
    """
    Check if a mask image has any white pixels (indicating AI modification)
    
    Args:
        mask_path (str): Path to the mask image
        
    Returns:
        bool: True if the mask has any white pixels (AI modified), False if all black (real image)
    """
    try:
        mask = Image.open(mask_path).convert('L')
        # Convert to numpy array for easier processing
        mask_array = np.array(mask)
        # Check if any pixel is not black (value > 0)
        return np.any(mask_array > 0)
    except Exception as e:
        print(f"Error processing mask {mask_path}: {e}")
        return None

def main():
    # Base directory for masks
    masks_dir = os.path.join('Yolov11', 'train', 'train', 'masks')
    
    # Get the first 200 mask files
    mask_files = [f"image_{i}.png" for i in range(200)]
    
    # Print header
    print(f"{'Image':<15} | {'Ground Truth':<15}")
    print("-" * 35)
    
    for mask_file in mask_files:
        mask_path = os.path.join(masks_dir, mask_file)
        
        # Skip if mask doesn't exist
        if not os.path.exists(mask_path):
            print(f"Mask file {mask_path} does not exist, skipping...")
            continue
        
        # Check mask to determine ground truth
        has_ai_from_mask = check_mask_has_ai(mask_path)
        if has_ai_from_mask is None:
            continue
        
        mask_result = "AI" if has_ai_from_mask else "Real"
        
        # Print results
        print(f"{mask_file:<15} | {mask_result:<15}")
    
if __name__ == "__main__":
    main() 