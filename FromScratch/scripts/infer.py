import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Import our modules
from models.unet import UNet
from dataset import get_test_loader
from utils import post_process_mask, save_predictions_to_csv


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
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
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
    
    return filenames, predicted_masks


def main(args):
    # Create output directories
    os.makedirs(args.output_masks_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    # Load trained weights
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    parser.add_argument("--test-images", type=str, default="../Yolov11/test/test/images",
                        help="Path to test images directory")
    parser.add_argument("--checkpoint", type=str, default="../models/best_model.pth",
                        help="Path to model checkpoint")
    
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
    parser.add_argument("--output-csv", type=str, default="../outputs/submission.csv",
                        help="Path to output CSV file")
    parser.add_argument("--save-masks", action="store_true",
                        help="Save predicted masks as images")
    parser.add_argument("--output-masks-dir", type=str, default="../outputs/predicted_masks",
                        help="Directory to save predicted mask images")
    
    args = parser.parse_args()
    
    main(args) 