import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random

# Import our modules
from models.unet_enhanced import UNetEnhanced
from scripts.dataset import get_loaders
from scripts.utils import CombinedLoss, dice_coefficient

# Simple progress indicator
def progress_print(current, total, message=""):
    """Simple progress indicator"""
    bar_length = 30
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{message} |{bar}| {current}/{total}', end='')
    if current == total:
        print()

# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EnhancedTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        device,
        checkpoint_dir="models",
        tensorboard_dir="logs",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        
        # Get total batches for progress reporting
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
            progress_print(
                batch_idx + 1, 
                total_batches, 
                f"Training Epoch {epoch} - Loss: {loss.item():.4f}, Dice: {dice:.4f}"
            )
        
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
        
        # Get total batches for progress reporting
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
                progress_print(
                    batch_idx + 1, 
                    total_batches, 
                    f"Validation Epoch {epoch} - Loss: {loss.item():.4f}, Dice: {dice:.4f}"
                )
        
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
            print(f"\nNew best model saved with Dice: {avg_dice:.4f}")
        
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
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            if self.scheduler is not None:
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                        "dice": val_dice,
                    },
                    os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                )
                print(f"Checkpoint saved at epoch {epoch}")
        
        # Record end time and print training duration
        end_time = time.time()
        train_duration = end_time - start_time
        hours, remainder = divmod(train_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Close TensorBoard writer
        self.writer.close()


def main(args):
    # Set random seeds
    seed_everything(args.seed)
    
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize enhanced U-Net model
    model = UNetEnhanced(
        n_channels=3, 
        n_classes=1, 
        bilinear=True, 
        dropout_p=args.dropout
    )
    model = model.to(device)
    print(f"Model initialized: UNetEnhanced with dropout={args.dropout}")
    
    # Print model parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {param_count:,} trainable parameters")
    
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
    
    print(f"Training data: {len(train_loader.dataset)} images")
    if val_loader:
        print(f"Validation data: {len(val_loader.dataset)} images")
    
    # Initialize loss function
    loss_fn = CombinedLoss(dice_weight=args.dice_weight, bce_weight=args.bce_weight)
    
    # Initialize optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:  # Default to AdamW
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Initialize learning rate scheduler
    if args.scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.scheduler.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.scheduler.lower() == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_gamma, patience=args.patience, 
            min_lr=args.min_lr, verbose=True
        )
    else:
        scheduler = None
    
    # Initialize trainer
    trainer = EnhancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir=args.tensorboard_dir,
    )
    
    # Train model
    trainer.train(num_epochs=args.epochs)
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Enhanced U-Net for AI Manipulation Detection")
    
    # Data paths
    parser.add_argument("--train-images", type=str, default="../train/train/images",
                        help="Path to training images directory")
    parser.add_argument("--train-masks", type=str, default="../train/train/masks",
                        help="Path to training masks directory")
    parser.add_argument("--val-images", type=str, default="data/val/images",
                        help="Path to validation images directory")
    parser.add_argument("--val-masks", type=str, default="data/val/masks",
                        help="Path to validation masks directory")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Initial learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay (L2 penalty) for optimizer")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability for model")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw"],
                        help="Optimizer type (adam or adamw)")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Size to resize images to")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Learning rate scheduler
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["step", "cosine", "reduce", "none"],
                        help="Learning rate scheduler type")
    parser.add_argument("--lr-step-size", type=int, default=10,
                        help="Step size for StepLR scheduler")
    parser.add_argument("--lr-gamma", type=float, default=0.1,
                        help="Gamma for StepLR and ReduceLROnPlateau schedulers")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Minimum learning rate for CosineAnnealingLR and ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for ReduceLROnPlateau scheduler")
    
    # Loss function weights
    parser.add_argument("--dice-weight", type=float, default=0.5,
                        help="Weight for Dice loss component")
    parser.add_argument("--bce-weight", type=float, default=0.5,
                        help="Weight for BCE loss component")
    
    # Output directories
    parser.add_argument("--checkpoint-dir", type=str, default="models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--tensorboard-dir", type=str, default="logs",
                        help="Directory for TensorBoard logs")
    
    args = parser.parse_args()
    
    main(args) 