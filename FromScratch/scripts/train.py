import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# Remove tqdm dependency
# from tqdm import tqdm
import random

# Import our modules
from models.unet import UNet
from dataset import get_loaders
from utils import CombinedLoss, dice_coefficient

# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Simple progress bar replacement
def progress_print(current, total, message="", metrics=None):
    """Simple progress indicator"""
    bar_length = 30
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    metrics_str = ""
    if metrics:
        metrics_str = " | " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    print(f'\r{message} |{bar}| {current}/{total}{metrics_str}', end='')
    if current == total:
        print()


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        checkpoint_dir="models",
        tensorboard_dir="logs",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(tensorboard_dir)
        
        # Initialize best validation metric
        self.best_dice = 0.0
    
    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch
        """
        self.model.train()
        
        # Initialize metrics
        epoch_loss = 0.0
        epoch_dice = 0.0
        
        # Progress bar replacement
        print(f"Training Epoch {epoch}")
        
        # Get total batch count
        total_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Get data
            images = batch["image"].to(self.device)
            masks = batch["mask"].unsqueeze(1).to(self.device)  # Add channel dimension
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Calculate Dice for monitoring (using sigmoid activation)
            with torch.no_grad():
                preds = torch.sigmoid(outputs) > 0.5
                dice = dice_coefficient(preds.float(), masks)
                epoch_dice += dice
            
            # Update progress bar
            progress_print(batch_idx + 1, total_batches, "Training", {"loss": loss.item(), "dice": dice})
        
        # Calculate average metrics
        avg_loss = epoch_loss / len(self.train_loader)
        avg_dice = epoch_dice / len(self.train_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        self.writer.add_scalar("Dice/train", avg_dice, epoch)
        
        return avg_loss, avg_dice
    
    def validate(self, epoch):
        """
        Validate the model on the validation set
        """
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        
        # Initialize metrics
        val_loss = 0.0
        val_dice = 0.0
        
        # Print a header
        print(f"Validation Epoch {epoch}")
        
        # Get total batch count
        total_batches = len(self.val_loader)
        
        # Disable gradients
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Get data
                images = batch["image"].to(self.device)
                masks = batch["mask"].unsqueeze(1).to(self.device)  # Add channel dimension
                
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                
                # Update metrics
                val_loss += loss.item()
                
                # Calculate Dice
                preds = torch.sigmoid(outputs) > 0.5
                dice = dice_coefficient(preds.float(), masks)
                val_dice += dice
                
                # Update progress bar
                progress_print(batch_idx + 1, total_batches, "Validation", {"loss": loss.item(), "dice": dice})
        
        # Calculate average metrics
        avg_loss = val_loss / len(self.val_loader)
        avg_dice = val_dice / len(self.val_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar("Loss/val", avg_loss, epoch)
        self.writer.add_scalar("Dice/val", avg_dice, epoch)
        
        # Save best model
        if avg_dice > self.best_dice:
            self.best_dice = avg_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "dice": avg_dice,
                },
                os.path.join(self.checkpoint_dir, "best_model.pth")
            )
            print(f"New best model saved with Dice: {avg_dice:.4f}")
        
        return avg_loss, avg_dice
    
    def train(self, num_epochs):
        """
        Train the model for multiple epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        # Record start time
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Print epoch header
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train one epoch
            train_loss, train_dice = self.train_one_epoch(epoch)
            
            # Validate
            val_loss, val_dice = self.validate(epoch)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "dice": val_dice,
                    },
                    os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                )
        
        # Record end time and print training duration
        end_time = time.time()
        train_duration = end_time - start_time
        print(f"\nTraining completed in {train_duration:.2f} seconds")
        
        # Close TensorBoard writer
        self.writer.close()


def main(args):
    # Set random seeds
    seed_everything(args.seed)
    
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize U-Net model
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model = model.to(device)
    
    # Create data loaders
    train_loader, val_loader = get_loaders(
        train_images_dir=args.train_images,
        train_masks_dir=args.train_masks,
        val_images_dir=args.val_images,
        val_masks_dir=args.val_masks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        img_size=args.img_size,
    )
    
    # Initialize loss function
    loss_fn = CombinedLoss(dice_weight=args.dice_weight, bce_weight=args.bce_weight)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir,
    )
    
    # Train model
    trainer.train(num_epochs=args.epochs)
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for AI Manipulation Detection")
    
    # Data paths
    parser.add_argument("--train-images", type=str, default="../Yolov11/train/train/images",
                        help="Path to training images directory")
    parser.add_argument("--train-masks", type=str, default="../Yolov11/train/train/masks",
                        help="Path to training masks directory")
    parser.add_argument("--val-images", type=str, default=None,
                        help="Path to validation images directory (optional)")
    parser.add_argument("--val-masks", type=str, default=None,
                        help="Path to validation masks directory (optional)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Size to resize images to")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Loss function weights
    parser.add_argument("--dice-weight", type=float, default=0.5,
                        help="Weight for Dice loss component")
    parser.add_argument("--bce-weight", type=float, default=0.5,
                        help="Weight for BCE loss component")
    
    # Output directories
    parser.add_argument("--checkpoint-dir", type=str, default="../models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--tensorboard-dir", type=str, default="../logs",
                        help="Directory for TensorBoard logs")
    
    args = parser.parse_args()
    
    main(args) 