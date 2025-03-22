import os
import argparse
import numpy as np
import torch
# Remove tqdm dependency
# from tqdm import tqdm

# Import our modules
from models.unet import UNet
try:
    from models.unet_enhanced import UNetEnhanced
    HAS_ENHANCED_MODEL = True
except ImportError:
    HAS_ENHANCED_MODEL = False
    
from dataset import get_test_loader
from utils import post_process_mask, save_predictions_to_csv


# Simple progress indicator
def progress_print(current, total, message=""):
    """Simple progress indicator"""
    bar_length = 30
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{message} |{bar}| {current}/{total}', end='')
    if current == total:
        print()


def predict_masks(model, test_loader, device, threshold=0.5, post_process=True, min_size=30):
    """
    Generate predictions on the test set
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run inference on
        threshold (float): Threshold for binary prediction
        post_process (bool): Whether to apply post-processing to masks
        min_size (int): Minimum size of regions to keep during post-processing
        
    Returns:
        tuple: Lists of filenames and predicted masks
    """
    model.eval()
    
    filenames = []
    predicted_masks = []
    
    print("Generating predictions...")
    total_batches = len(test_loader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            batch_filenames = batch["filename"]
            
            # Forward pass
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            # Process each image in the batch
            for i, filename in enumerate(batch_filenames):
                mask = preds[i, 0]  # Get mask (remove batch and channel dimensions)
                
                # Apply post-processing if enabled
                if post_process:
                    mask = post_process_mask(mask, threshold, min_size)
                else:
                    mask = (mask > threshold).astype(np.uint8)
                
                # Save to lists
                filenames.append(filename)
                predicted_masks.append(mask)
                
                # Optionally save individual mask images
                if args.save_masks:
                    # Convert to binary image and save
                    mask_image = (mask * 255).astype(np.uint8)
                    from PIL import Image
                    mask_img = Image.fromarray(mask_image)
                    mask_img.save(os.path.join(args.output_masks_dir, filename))
            
            # Update progress
            progress_print(batch_idx + 1, total_batches, "Processing batches")
    
    return filenames, predicted_masks


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
    
    # Check for enhanced model keys (inc.0.double_conv or inc.channel_attention)
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
    
    # Default to standard if can't determine
    print("WARNING: Could not definitively detect model type from checkpoint. Defaulting to standard UNet.")
    return 'standard'


def main(args):
    # Create output directories
    os.makedirs(args.output_masks_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint first to determine model type
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Auto-detect model type from checkpoint if not explicitly specified or if mismatch
    if args.auto_detect_model:
        detected_model_type = detect_model_type(checkpoint)
        if args.model_type != detected_model_type:
            print(f"Auto-detected model type '{detected_model_type}' differs from specified '{args.model_type}'")
            print(f"Using auto-detected model type: {detected_model_type}")
            args.model_type = detected_model_type
    
    # Initialize model based on detected or specified type
    if args.model_type == 'enhanced' and HAS_ENHANCED_MODEL:
        print("Using Enhanced UNet model with attention")
        model = UNetEnhanced(n_channels=3, n_classes=1, bilinear=True, dropout_p=args.dropout)
    else:
        if args.model_type == 'enhanced' and not HAS_ENHANCED_MODEL:
            print("Enhanced model requested but not available. Falling back to standard UNet.")
        else:
            print("Using standard UNet model")
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    # Load weights
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model weights loaded successfully")
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        if args.auto_detect_model:
            print("Error despite auto-detection. This might indicate a custom or modified model architecture.")
        else:
            print("Try using --auto-detect-model flag to automatically detect the correct model architecture")
        return
    
    model = model.to(device)
    
    # Print model information
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Model Dice score: {checkpoint.get('dice', 'unknown')}")
    
    # Create test data loader
    test_loader = get_test_loader(
        test_images_dir=args.test_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        img_size=args.img_size,
    )
    
    # Generate predictions
    print("Generating predictions...")
    filenames, predicted_masks = predict_masks(
        model=model,
        test_loader=test_loader,
        device=device,
        threshold=args.threshold,
        post_process=args.post_process,
        min_size=args.min_size,
    )
    
    # Save predictions to CSV
    print(f"Saving predictions to {args.output_csv}")
    save_predictions_to_csv(
        image_filenames=filenames,
        masks=predicted_masks,
        output_file=args.output_csv,
    )
    
    print("Inference completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for AI Manipulation Detection")
    
    # Data paths
    parser.add_argument("--test-images", type=str, default="test/images",
                        help="Path to test images directory")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth",
                        help="Path to model checkpoint")
    
    # Model type
    parser.add_argument("--model-type", type=str, default="standard",
                        choices=["standard", "enhanced"],
                        help="Type of UNet model to use (standard or enhanced)")
    parser.add_argument("--auto-detect-model", action="store_true",
                        help="Automatically detect model type from checkpoint")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability for enhanced model (ignored for standard model)")
    
    # Inference parameters
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Size to resize images to")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary prediction")
    
    # Post-processing
    parser.add_argument("--post-process", action="store_true",
                        help="Apply post-processing to predicted masks")
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