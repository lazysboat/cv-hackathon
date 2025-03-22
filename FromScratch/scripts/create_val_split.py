import os
import argparse
import random
import shutil
from tqdm import tqdm

def create_validation_split(
    train_images_dir,
    train_masks_dir,
    val_images_dir,
    val_masks_dir,
    val_ratio=0.2,
    random_seed=42
):
    """
    Create a validation split from training data
    
    Args:
        train_images_dir (str): Path to training images directory
        train_masks_dir (str): Path to training masks directory
        val_images_dir (str): Path to output validation images directory
        val_masks_dir (str): Path to output validation masks directory
        val_ratio (float): Proportion of data to use for validation (0.0-1.0)
        random_seed (int): Random seed for reproducibility
    """
    # Set random seed
    random.seed(random_seed)
    
    # Create output directories
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.png')]
    
    # Determine number of validation samples
    num_val = int(len(image_files) * val_ratio)
    print(f"Total images: {len(image_files)}")
    print(f"Validation split ratio: {val_ratio}")
    print(f"Number of validation samples: {num_val}")
    
    # Randomly select validation samples
    val_files = random.sample(image_files, num_val)
    
    # Move validation files
    print("Creating validation split...")
    for filename in tqdm(val_files):
        # Move image
        src_img = os.path.join(train_images_dir, filename)
        dst_img = os.path.join(val_images_dir, filename)
        shutil.copy(src_img, dst_img)
        
        # Move mask
        src_mask = os.path.join(train_masks_dir, filename)
        dst_mask = os.path.join(val_masks_dir, filename)
        shutil.copy(src_mask, dst_mask)
    
    print(f"Validation split created: {num_val} samples")
    print(f"Remaining training samples: {len(image_files) - num_val}")


def main(args):
    create_validation_split(
        train_images_dir=args.train_images,
        train_masks_dir=args.train_masks,
        val_images_dir=args.val_images,
        val_masks_dir=args.val_masks,
        val_ratio=args.val_ratio,
        random_seed=args.seed
    )
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a validation split from training data")
    
    # Data paths
    parser.add_argument("--train-images", type=str, default="../Yolov11/train/train/images",
                        help="Path to training images directory")
    parser.add_argument("--train-masks", type=str, default="../Yolov11/train/train/masks",
                        help="Path to training masks directory")
    parser.add_argument("--val-images", type=str, default="../data/val/images",
                        help="Path to output validation images directory")
    parser.add_argument("--val-masks", type=str, default="../data/val/masks",
                        help="Path to output validation masks directory")
    
    # Split parameters
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Proportion of data to use for validation (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    main(args) 