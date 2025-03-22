import os
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

def test_model(image_path=None, model_path=None, save_dir=None, threshold=0.5):
    """
    Test the trained YOLO segmentation model on a single image or directory of images.
    
    Args:
        image_path: Path to an image or directory of images to test
        model_path: Path to the trained model weights
        save_dir: Directory to save visualization results
        threshold: Confidence threshold for segmentation (0-1)
    """
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set default model path if not provided
    if model_path is None:
        # Look for best model in default location
        default_model_paths = [
            os.path.join(script_dir, 'runs', 'segment', 'train', 'weights', 'best.pt'),
            os.path.join(script_dir, 'runs', 'segment', 'train', 'weights', 'last.pt'),
            os.path.join('runs', 'segment', 'train', 'weights', 'best.pt'),
            os.path.join('runs', 'segment', 'train', 'weights', 'last.pt'),
        ]
        
        for path in default_model_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Using model: {path}")
                break
        
        if model_path is None:
            print("Error: Could not find trained model weights.")
            print("Please specify model path with --model")
            return
    
    # Set default image path if not provided
    if image_path is None:
        # Look for test images
        test_paths = [
            os.path.join(script_dir, 'train', 'test', 'images'),
            os.path.join(script_dir, 'test', 'images')
        ]
        
        for path in test_paths:
            if os.path.exists(path):
                image_path = path
                print(f"Using test directory: {path}")
                break
        
        if image_path is None:
            print("Error: No image path provided and no default test directory found.")
            print("Please specify an image or directory with --image")
            return
    
    # Create save directory if not exists
    if save_dir is None:
        save_dir = os.path.join(script_dir, 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the model
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Check if image path is a file or directory
    if os.path.isfile(image_path):
        image_paths = [image_path]
    else:
        # Get all image files from directory
        image_paths = []
        for ext in ['.png', '.jpg', '.jpeg']:
            image_paths.extend(
                [os.path.join(image_path, f) for f in os.listdir(image_path) 
                 if f.lower().endswith(ext)]
            )
        
        # Limit to first 10 images if testing a directory
        if len(image_paths) > 10:
            print(f"Found {len(image_paths)} images. Testing first 10 for preview.")
            image_paths = image_paths[:10]
        else:
            print(f"Found {len(image_paths)} images to test.")
    
    # Process each image
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        # Run prediction
        results = model.predict(
            source=img_path,
            save=False,
            conf=threshold,
            retina_masks=True,  # High-resolution masks
            verbose=False
        )
        
        # Get original image
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Create figure for visualization
        plt.figure(figsize=(18, 6))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title("Original Image")
        plt.axis('off')
        
        # Plot predicted mask
        for r in results:
            if r.masks is not None:
                # Get the mask
                mask = r.masks.data.cpu().numpy()[0]
                
                # Plot mask alone
                plt.subplot(1, 3, 2)
                plt.imshow(mask, cmap='gray')
                plt.title("Predicted Mask")
                plt.axis('off')
                
                # Create overlay image
                overlay = original_img.copy()
                # Create colored mask (red for manipulated regions)
                colored_mask = np.zeros_like(original_img)
                colored_mask[mask > 0.5] = [255, 0, 0]  # Red for manipulated regions
                
                # Blend with 50% opacity
                overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
                
                plt.subplot(1, 3, 3)
                plt.imshow(overlay)
                plt.title("Overlay (Red = AI Manipulated)")
                plt.axis('off')
            else:
                # No masks detected
                plt.subplot(1, 3, 2)
                plt.imshow(np.zeros_like(original_img[:,:,0]), cmap='gray')
                plt.title("No Manipulation Detected")
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(original_img)
                plt.title("Original (No Manipulation Detected)")
                plt.axis('off')
        
        # Save the visualization
        save_path = os.path.join(save_dir, f"pred_{filename}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"  Saved prediction to {save_path}")
    
    print(f"\nAll predictions saved to {save_dir}")
    print("\nTo visualize more images, run:")
    print(f"python test_model.py --image path/to/image.jpg --model {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Test AI Manipulation Detection model")
    parser.add_argument('--image', type=str, default=None, 
                        help='Path to an image or directory of images to test')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the trained model weights')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save visualization results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for segmentation (0-1)')
    
    args = parser.parse_args()
    
    test_model(
        image_path=args.image,
        model_path=args.model,
        save_dir=args.save_dir,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main() 