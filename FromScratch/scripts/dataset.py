import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AIManipulationDataset(Dataset):
    """
    Dataset class for AI Manipulation Detection
    """
    def __init__(self, images_dir, masks_dir=None, transform=None, is_test=False):
        """
        Args:
            images_dir (str): Path to the images directory
            masks_dir (str, optional): Path to the masks directory. None for test data.
            transform (albumentations.Compose, optional): Transformations to apply
            is_test (bool): Whether this is a test dataset (no ground truth masks)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.is_test = is_test
        
        # Get list of image files
        self.image_files = sorted(
            [f for f in os.listdir(images_dir) if f.endswith('.png') or f.endswith('.jpg')]
        )
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # For test data, we don't have masks
        if self.is_test:
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            
            return {
                "image": image,
                "filename": self.image_files[idx]
            }
        
        # Load mask for training data
        mask_path = os.path.join(self.masks_dir, self.image_files[idx])
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Normalize mask to binary (0 or 1)
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)
        
        # Apply transformations if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return {
            "image": image,
            "mask": mask,
            "filename": self.image_files[idx]
        }


def get_train_transform(img_size=256):
    """
    Get training transformations with augmentations
    """
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.3),
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_valid_transform(img_size=256):
    """
    Get validation transformations (no augmentations)
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_test_transform(img_size=256):
    """
    Get test transformations (same as validation)
    """
    return get_valid_transform(img_size)


def get_loaders(
    train_images_dir,
    train_masks_dir,
    val_images_dir=None,
    val_masks_dir=None,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    img_size=256,
):
    """
    Create train and validation data loaders
    
    Args:
        train_images_dir (str): Directory with training images
        train_masks_dir (str): Directory with training masks
        val_images_dir (str, optional): Directory with validation images. 
                                      If None, validation loader will be None.
        val_masks_dir (str, optional): Directory with validation masks
        batch_size (int): Batch size
        num_workers (int): Number of workers for DataLoader
        pin_memory (bool): Whether to pin memory in DataLoader
        img_size (int): Size to resize images to
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_ds = AIManipulationDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        transform=get_train_transform(img_size),
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    val_loader = None
    if val_images_dir and val_masks_dir:
        val_ds = AIManipulationDataset(
            images_dir=val_images_dir,
            masks_dir=val_masks_dir,
            transform=get_valid_transform(img_size),
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
    
    return train_loader, val_loader


def get_test_loader(
    test_images_dir,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    img_size=256,
):
    """
    Create test data loader
    
    Args:
        test_images_dir (str): Directory with test images
        batch_size (int): Batch size
        num_workers (int): Number of workers for DataLoader
        pin_memory (bool): Whether to pin memory in DataLoader
        img_size (int): Size to resize images to
        
    Returns:
        DataLoader: Test data loader
    """
    test_ds = AIManipulationDataset(
        images_dir=test_images_dir,
        masks_dir=None,  # No masks for test data
        transform=get_test_transform(img_size),
        is_test=True,
    )
    
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    ) 