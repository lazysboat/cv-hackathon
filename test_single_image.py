import os
import sys
import numpy as np
from PIL import Image
import argparse

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
    parser = argparse.ArgumentParser(description='Test AI detector on a range of images')
    parser.add_argument('start', type=int, help='Starting image number (e.g., 0 for image_0.png)')
    parser.add_argument('end', type=int, help='Ending image number (inclusive)', nargs='?', default=None)
    args = parser.parse_args()
    
    # Base directories for images and masks
    images_dir = os.path.join('Yolov11', 'train', 'train', 'images')
    masks_dir = os.path.join('Yolov11', 'train', 'train', 'masks')
    
    # If end is not provided, just test a single image
    if args.end is None:
        image_numbers = [args.start]
    else:
        # Test a range of images
        image_numbers = range(args.start, args.end + 1)
    
    # Results table header
    print(f"{'Image':<15} | {'AI Detection':<15} | {'Confidence':<15} | {'Mask Truth':<15} | {'Match':<10}")
    print("-" * 80)
    
    correct_count = 0
    total_count = 0
    
    for img_num in image_numbers:
        image_file = f"image_{img_num}.png"
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, image_file)
        
        # Skip if image or mask doesn't exist
        if not os.path.exists(image_path):
            print(f"Image file {image_path} does not exist!")
            continue
        if not os.path.exists(mask_path):
            print(f"Mask file {mask_path} does not exist!")
            continue
        
        print(f"Testing AI detector on {image_file}...")
        
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