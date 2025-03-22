import os
import argparse
import subprocess
import sys

# Use python3 if on Linux/Unix, python otherwise
PYTHON_CMD = 'python3' if sys.platform in ('linux', 'linux2', 'darwin') else 'python'

def setup_directories():
    """Set up required directories if they don't exist"""
    dirs = [
        "data/val/images",
        "data/val/masks",
        "models",
        "outputs/predicted_masks",
        "logs"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")


def create_val_split(args):
    """Create validation split from training data"""
    print("\n=== Creating Validation Split ===")
    cmd = [
        PYTHON_CMD, "scripts/create_val_split.py",
        "--train-images", args.train_images,
        "--train-masks", args.train_masks,
        "--val-images", "data/val/images",
        "--val-masks", "data/val/masks",
        "--val-ratio", str(args.val_ratio),
        "--seed", str(args.seed)
    ]
    subprocess.run(cmd)


def train_model(args):
    """Train the model"""
    print("\n=== Training Model ===")
    
    # Use enhanced model if requested
    if args.enhanced:
        cmd = [
            PYTHON_CMD, "train_enhanced.py",
            "--train-images", args.train_images,
            "--train-masks", args.train_masks,
            "--val-images", "data/val/images",
            "--val-masks", "data/val/masks",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--dropout", str(args.dropout),
            "--weight-decay", str(args.weight_decay),
            "--optimizer", args.optimizer,
            "--img-size", str(args.img_size),
            "--num-workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--scheduler", args.scheduler,
            "--dice-weight", str(args.dice_weight),
            "--bce-weight", str(args.bce_weight),
            "--checkpoint-dir", "models",
            "--tensorboard-dir", "logs"
        ]
    else:
        # Use original training script
        cmd = [
            PYTHON_CMD, "scripts/train.py",
            "--train-images", args.train_images,
            "--train-masks", args.train_masks,
            "--val-images", "data/val/images",
            "--val-masks", "data/val/masks",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--img-size", str(args.img_size),
            "--num-workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--dice-weight", str(args.dice_weight),
            "--bce-weight", str(args.bce_weight),
            "--checkpoint-dir", "models",
            "--tensorboard-dir", "logs"
        ]
    
    subprocess.run(cmd)


def run_inference(args):
    """Run inference on test set"""
    print("\n=== Running Inference ===")
    cmd = [
        PYTHON_CMD, "scripts/infer.py",
        "--test-images", args.test_images,
        "--checkpoint", "models/best_model.pth",
        "--batch-size", str(args.batch_size),
        "--img-size", str(args.img_size),
        "--num-workers", str(args.num_workers),
        "--threshold", str(args.threshold),
        "--output-csv", "outputs/submission.csv"
    ]
    
    # Pass model type if using enhanced model
    if args.enhanced:
        cmd.extend(["--model-type", "enhanced", "--dropout", str(args.dropout)])
    
    # Add auto-detect model flag if specified
    if args.auto_detect_model:
        cmd.append("--auto-detect-model")
    
    if args.post_process:
        cmd.append("--post-process")
        cmd.extend(["--min-size", str(args.min_size)])
    
    if args.save_masks:
        cmd.append("--save-masks")
        cmd.extend(["--output-masks-dir", "outputs/predicted_masks"])
    
    subprocess.run(cmd)


def main(args):
    # Set up directories
    setup_directories()
    
    # Create validation split if needed
    if args.create_val:
        create_val_split(args)
    
    # Train model if requested
    if args.train:
        train_model(args)
    
    # Run inference if requested
    if args.inference:
        run_inference(args)
    
    print("\nProcess completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Manipulation Detection - Training and Inference Pipeline")
    
    # Operation mode
    parser.add_argument("--create-val", action="store_true",
                        help="Create validation split from training data")
    parser.add_argument("--train", action="store_true",
                        help="Train the model")
    parser.add_argument("--inference", action="store_true",
                        help="Run inference on test set")
    parser.add_argument("--enhanced", action="store_true",
                        help="Use the enhanced U-Net model with attention and residual connections")
    parser.add_argument("--auto-detect-model", action="store_true",
                        help="Automatically detect model type from checkpoint during inference")
    
    # Data paths
    parser.add_argument("--train-images", type=str, default="../train/train/images",
                        help="Path to training images directory")
    parser.add_argument("--train-masks", type=str, default="../train/train/masks",
                        help="Path to training masks directory")
    parser.add_argument("--test-images", type=str, default="../test/test/images",
                        help="Path to test images directory")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training and inference")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Size to resize images to")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Proportion of data to use for validation (0.0-1.0)")
    
    # Enhanced model parameters
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability for enhanced model")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw"],
                        help="Optimizer type (adam or adamw)")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["step", "cosine", "reduce", "none"],
                        help="Learning rate scheduler type")
    
    # Loss function weights
    parser.add_argument("--dice-weight", type=float, default=0.5,
                        help="Weight for Dice loss component")
    parser.add_argument("--bce-weight", type=float, default=0.5,
                        help="Weight for BCE loss component")
    
    # Inference parameters
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary prediction")
    parser.add_argument("--post-process", action="store_true",
                        help="Apply post-processing to predicted masks")
    parser.add_argument("--min-size", type=int, default=30,
                        help="Minimum size of regions to keep during post-processing")
    parser.add_argument("--save-masks", action="store_true",
                        help="Save predicted masks as images")
    
    args = parser.parse_args()
    
    # If no operation specified, print help
    if not any([args.create_val, args.train, args.inference]):
        parser.print_help()
        print("\nNo operation specified. Please use --create-val, --train, or --inference.")
        exit(0)
    
    main(args) 