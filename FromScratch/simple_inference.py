import os
import argparse
import numpy as np
import torch
from PIL import Image
import glob
from torchvision import transforms
try:
    import cv2
    from scipy import ndimage
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not found. Advanced post-processing will be limited.")

# Import our modules
try:
    from models.unet import UNet
except ImportError:
    print("WARNING: Could not import UNet model. Creating a simple version for loading weights.")
    # Create a simplified UNet model that matches the architecture
    import torch.nn as nn
    
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            return self.double_conv(x)
    
    class Down(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
        
        def forward(self, x):
            return self.maxpool_conv(x)
    
    class Up(nn.Module):
        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
            self.conv = DoubleConv(in_channels, out_channels)
        
        def forward(self, x1, x2):
            x1 = self.up(x1)
            # Padding
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            # Concatenate
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
    
    class OutConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        def forward(self, x):
            return self.conv(x)
    
    class UNet(nn.Module):
        def __init__(self, n_channels=3, n_classes=1, bilinear=True):
            super(UNet, self).__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.bilinear = bilinear
            
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            self.up1 = Up(1024, 512 // factor, bilinear)
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            self.up4 = Up(128, 64, bilinear)
            self.outc = OutConv(64, n_classes)
        
        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits

try:
    from models.unet_enhanced import UNetEnhanced
    HAS_ENHANCED_MODEL = True
except ImportError:
    HAS_ENHANCED_MODEL = False
    print("Enhanced UNet model not available. Will try to use standard UNet model.")


def detect_model_type(checkpoint):
    """
    Detect whether the checkpoint was trained with standard UNet or enhanced UNet
    by examining the keys in the state dictionary
    
    Args:
        checkpoint (dict): Loaded model checkpoint
        
    Returns:
        str: 'standard' or 'enhanced'
    """
    keys = list(checkpoint["model_state_dict"].keys())
    
    # Check for enhanced model keys
    enhanced_indicators = ['inc.0.', 'channel_attention', 'spatial_attention', 'residual']
    standard_indicators = ['inc.double_conv.0.weight', 'down1.maxpool_conv.1.double_conv.0.weight']
    
    # Check if any enhanced model indicators are present
    for key in keys:
        for indicator in enhanced_indicators:
            if indicator in key:
                return 'enhanced'
    
    # Check for standard model indicators
    for key in keys:
        for indicator in standard_indicators:
            if key == indicator:
                return 'standard'
    
    # If both checks fail, look at key structure
    if 'inc.double_conv.0.weight' in keys:
        return 'standard'
    elif any('inc.0' in key for key in keys):
        return 'enhanced'
    
    # Default to standard if can't determine
    print("WARNING: Could not definitively detect model type from checkpoint. Defaulting to standard UNet.")
    return 'standard'


def load_image(file_path, img_size=256):
    """Load and preprocess an image file"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(file_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def post_process_mask(mask, threshold=0.5, min_size=30, advanced=False):
    """
    Apply post-processing to a predicted mask
    
    Args:
        mask (numpy.ndarray): Predicted mask (probability map)
        threshold (float): Threshold for binary prediction
        min_size (int): Minimum size of regions to keep
        advanced (bool): Whether to use advanced post-processing techniques
        
    Returns:
        numpy.ndarray: Post-processed binary mask
    """
    # Basic thresholding
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Return early if no advanced processing or OpenCV is not available
    if not advanced or not HAS_OPENCV:
        return binary_mask
    
    # Advanced processing techniques
    
    # 1. Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    # Opening (erosion followed by dilation) to remove small noise
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing (dilation followed by erosion) to fill small holes
    processed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 2. Remove small connected components
    if min_size > 0:
        # Label connected components
        labeled_mask, num_features = ndimage.label(processed_mask)
        # Calculate size of each component
        component_sizes = np.bincount(labeled_mask.ravel())
        # Set small components to 0 (background)
        too_small = component_sizes < min_size
        too_small[0] = False  # Don't remove the background
        processed_mask = processed_mask.copy()  # Make a copy to avoid modifying the original
        processed_mask[np.isin(labeled_mask, np.where(too_small)[0])] = 0
    
    # 3. Fill holes in the mask (optional)
    if np.any(processed_mask):  # Only if there are any positive pixels
        # Fill holes (background regions surrounded by foreground)
        filled_mask = ndimage.binary_fill_holes(processed_mask).astype(np.uint8)
        
        # Only use filled mask if it doesn't increase the mask size too much
        filled_area = np.sum(filled_mask)
        original_area = np.sum(processed_mask)
        
        # If filling holes increases area by more than 50%, use the original
        if filled_area <= original_area * 1.5:
            processed_mask = filled_mask
    
    # 4. Boundary refinement using active contours (Snake algorithm)
    # This is computationally expensive, so only use if the mask has reasonable size
    if np.sum(processed_mask) > 100 and np.sum(processed_mask) < mask.size * 0.5:
        try:
            # Convert probability map to grayscale for edge detection
            gray_mask = (mask * 255).astype(np.uint8)
            # Find initial contour
            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Only refine if we found contours
            if contours and len(contours) > 0:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Only refine substantial contours
                if cv2.contourArea(largest_contour) > 100:
                    # Create a mask with only the largest contour
                    contour_mask = np.zeros_like(processed_mask)
                    cv2.drawContours(contour_mask, [largest_contour], 0, 1, -1)
                    
                    # Dilate slightly to ensure the active contour has room to work
                    dilated_contour = cv2.dilate(contour_mask, kernel, iterations=2)
                    
                    # Use the dilated contour as a mask for the original probabilities
                    # This helps the snake algorithm focus on the boundary
                    edge_mask = gray_mask * dilated_contour
                    
                    # Apply active contour (snake) algorithm
                    refined_contour = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                    if refined_contour and len(refined_contour) > 0:
                        refined_contour = max(refined_contour, key=cv2.contourArea)
                        if cv2.contourArea(refined_contour) > 100:
                            # Create a mask with the refined contour
                            refined_mask = np.zeros_like(processed_mask)
                            cv2.drawContours(refined_mask, [refined_contour], 0, 1, -1)
                            
                            # Combine with processed mask (use the union)
                            processed_mask = np.maximum(processed_mask, refined_mask)
        except Exception as e:
            print(f"Warning: Error in boundary refinement: {e}")
            # Fallback to the processed mask without refinement
    
    return processed_mask


def advanced_ensemble_post_processing(mask, advanced_config=None):
    """
    Apply an ensemble of post-processing techniques with different parameters
    and combine the results
    
    Args:
        mask (numpy.ndarray): Predicted mask (probability map)
        advanced_config (dict): Configuration for post-processing
        
    Returns:
        numpy.ndarray: Post-processed binary mask
    """
    if advanced_config is None:
        # Default configuration
        advanced_config = {
            'thresholds': [0.3, 0.4, 0.5, 0.6, 0.7],
            'min_sizes': [0, 30, 50],
            'weights': [1, 1, 1.5, 1, 0.8]  # Weight each threshold result
        }
    
    # Apply different thresholds and min_sizes
    processed_masks = []
    
    for threshold, weight in zip(advanced_config['thresholds'], advanced_config['weights']):
        for min_size in advanced_config['min_sizes']:
            processed = post_process_mask(mask, threshold, min_size, advanced=True)
            # Add the processed mask with its weight
            processed_masks.append((processed, weight))
    
    # Combine the processed masks with weights
    combined_mask = np.zeros_like(mask, dtype=float)
    total_weight = 0
    
    for processed, weight in processed_masks:
        combined_mask += processed * weight
        total_weight += weight
    
    # Normalize and convert to binary
    if total_weight > 0:
        combined_mask /= total_weight
    
    # Final thresholding (using 0.5 as we've already weighted the results)
    binary_mask = (combined_mask > 0.5).astype(np.uint8)
    
    return binary_mask


def save_predictions_to_csv(image_filenames, masks, output_file):
    """
    Save predicted masks to a CSV file in required format
    
    Args:
        image_filenames (list): List of image filenames
        masks (list): List of predicted masks
        output_file (str): Path to output CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Open CSV file for writing
    with open(output_file, 'w') as f:
        f.write('filename,rle\n')  # CSV header
        
        for filename, mask in zip(image_filenames, masks):
            # Convert mask to RLE format
            rle = mask2rle(mask)
            
            # Write to CSV
            f.write(f'{filename},{rle}\n')
    
    print(f"Saved predictions to {output_file}")


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def main(args):
    # Create output directories
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if args.save_masks:
        os.makedirs(args.output_masks_dir, exist_ok=True)
    
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Always use the enhanced model
    try:
        from models.unet_enhanced import UNetEnhanced
        print("Using Enhanced UNet model with attention")
        model = UNetEnhanced(n_channels=3, n_classes=1, bilinear=True, dropout_p=args.dropout)
    except ImportError:
        print("Enhanced model not available. Using standard UNet model")
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    # Load weights
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model weights loaded successfully")
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        print("Trying to adapt the state dictionary to match the model...")
        
        # Attempt to adapt state dict to model
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        
        # Map typical enhanced model keys to standard model keys
        for key, value in state_dict.items():
            if 'inc.0' in key and 'inc.double_conv' not in key:
                new_key = key.replace('inc.0', 'inc.double_conv')
                new_state_dict[new_key] = value
            elif 'down1.1' in key:
                new_key = key.replace('down1.1', 'down1.maxpool_conv.1')
                new_state_dict[new_key] = value
            elif 'down2.1' in key:
                new_key = key.replace('down2.1', 'down2.maxpool_conv.1')
                new_state_dict[new_key] = value
            elif 'down3.1' in key:
                new_key = key.replace('down3.1', 'down3.maxpool_conv.1')
                new_state_dict[new_key] = value
            elif 'down4.1' in key:
                new_key = key.replace('down4.1', 'down4.maxpool_conv.1')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("Adapted state dictionary loaded successfully with non-strict matching")
        except RuntimeError as e:
            print(f"Still couldn't load weights: {e}")
            print("Continuing with randomly initialized weights - results will be poor!")
    
    model = model.to(device)
    model.eval()
    
    # Find all test images
    test_images = glob.glob(os.path.join(args.test_images, '*.png'))
    print(f"Found {len(test_images)} test images")
    
    # Generate predictions
    print("Generating predictions...")
    filenames = []
    predicted_masks = []
    
    with torch.no_grad():
        for i, img_path in enumerate(test_images):
            # Print progress
            if i % 100 == 0:
                print(f"Processing image {i+1}/{len(test_images)}...")
            
            # Get filename from path
            filename = os.path.basename(img_path)
            filenames.append(filename)
            
            # Load and preprocess image
            image = load_image(img_path, args.img_size).to(device)
            
            # Generate prediction
            output = model(image)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0]  # Remove batch and channel dimensions
            
            # Apply post-processing
            if args.ensemble_post_process:
                # Apply ensemble of post-processing techniques
                mask = advanced_ensemble_post_processing(pred)
            elif args.advanced_post_process:
                # Apply advanced post-processing
                mask = post_process_mask(pred, args.threshold, args.min_size, advanced=True)
            elif args.post_process:
                # Apply basic post-processing
                mask = post_process_mask(pred, args.threshold, args.min_size, advanced=False)
            else:
                # Just threshold
                mask = (pred > args.threshold).astype(np.uint8)
            
            predicted_masks.append(mask)
            
            # Optionally save mask image
            if args.save_masks:
                mask_image = (mask * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_image)
                mask_img.save(os.path.join(args.output_masks_dir, filename))
    
    # Save predictions to CSV
    print(f"Saving predictions to {args.output_csv}")
    save_predictions_to_csv(filenames, predicted_masks, args.output_csv)
    
    print("Inference completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple inference for AI Manipulation Detection")
    
    # Data paths
    parser.add_argument("--test-images", type=str, default="../test/test/images",
                        help="Path to test images directory")
    parser.add_argument("--model-path", type=str, default="models/best_model.pth",
                        help="Path to model checkpoint")
    
    # Inference parameters
    parser.add_argument("--img-size", type=int, default=256,
                        help="Size to resize images to (must match training size)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary prediction")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability for enhanced model")
    
    # Post-processing
    parser.add_argument("--post-process", action="store_true",
                        help="Apply basic post-processing to predicted masks")
    parser.add_argument("--advanced-post-process", action="store_true",
                        help="Apply advanced post-processing techniques")
    parser.add_argument("--ensemble-post-process", action="store_true",
                        help="Apply ensemble of post-processing techniques")
    parser.add_argument("--min-size", type=int, default=30,
                        help="Minimum size of regions to keep during post-processing")
    
    # Output paths
    parser.add_argument("--output-csv", type=str, default="outputs/submission.csv",
                        help="Path to output CSV file")
    parser.add_argument("--save-masks", action="store_true",
                        help="Save predicted masks as images")
    parser.add_argument("--output-masks-dir", type=str, default="outputs/predicted_masks",
                        help="Directory to save predicted mask images")
    
    args = parser.parse_args()
    
    main(args) 