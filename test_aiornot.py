import os
import sys
import numpy as np
from PIL import Image

# Add the AiOrNot directory to the path so we can import the module
sys.path.append('AiOrNot')
from aiornot import classify_image

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
    # Base directories for images and masks
    images_dir = os.path.join('Yolov11', 'train', 'train', 'images')
    masks_dir = os.path.join('Yolov11', 'train', 'train', 'masks')
    
    # Get the first 200 image files
    image_files = [f"image_{i}.png" for i in range(200)]
    
    # Results table header
    print(f"{'Image':<15} | {'AI Detection':<15} | {'Confidence':<15} | {'Mask Truth':<15} | {'Match':<10}")
    print("-" * 80)
    
    correct_count = 0
    total_count = 0
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, image_file)
        
        # Skip if image or mask doesn't exist
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Missing file for {image_file}, skipping...")
            continue
        
        # Run AI detector on image
        ai_result, confidence = classify_image(image_path)
        
        # Check mask to determine ground truth
        has_ai_from_mask = check_mask_has_ai(mask_path)
        mask_result = "AI" if has_ai_from_mask else "Real"
        
        # Compare results
        match = (ai_result == "AI" and has_ai_from_mask) or (ai_result == "Real" and not has_ai_from_mask)
        match_str = "✓" if match else "✗"
        
        if match:
            correct_count += 1
        total_count += 1
        
        # Print results
        print(f"{image_file:<15} | {ai_result:<15} | {confidence:<15.2f}% | {mask_result:<15} | {match_str:<10}")
    
    # Print accuracy
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct_count}/{total_count} correct)")
    
if __name__ == "__main__":
    main() 