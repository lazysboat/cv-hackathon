import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        batch_size = probs.shape[0]
        
        # Flatten predictions and targets
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        # Calculate Dice score
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score
        
        return dice_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined BCE and Dice loss for binary segmentation
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        combined_loss = self.bce_weight * bce + self.dice_weight * dice
        return combined_loss


def dice_coefficient(y_pred, y_true, smooth=1.0):
    """
    Calculate Dice coefficient for evaluation
    
    Args:
        y_pred (torch.Tensor): Predicted mask (after sigmoid, shape: B x 1 x H x W)
        y_true (torch.Tensor): Ground truth mask (shape: B x 1 x H x W)
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: Dice coefficient
    """
    # Flatten tensors
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    dice = (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    
    return dice.item()


def mask_to_rle(mask):
    """
    Convert a binary mask to Run-Length Encoding (RLE)
    
    Args:
        mask (numpy.ndarray): Binary mask of shape (H, W) with values 0 or 1
        
    Returns:
        str: RLE encoding as a string
    """
    # Flatten the mask and ensure it's binary
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)


def rle_to_mask(rle, shape):
    """
    Convert RLE encoding to binary mask
    
    Args:
        rle (str): RLE encoding as a string
        shape (tuple): Shape of the mask (H, W)
        
    Returns:
        numpy.ndarray: Binary mask of shape (H, W) with values 0 or 1
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape)


def save_predictions_to_csv(image_filenames, masks, output_file):
    """
    Save predictions to CSV file in the required format
    
    Args:
        image_filenames (list): List of image filenames
        masks (list): List of binary masks as numpy arrays
        output_file (str): Path to output CSV file
    """
    with open(output_file, 'w') as f:
        f.write('image_id,rle_mask\n')
        for filename, mask in zip(image_filenames, masks):
            rle = mask_to_rle(mask)
            f.write(f'{filename},{rle}\n')
            
            
def post_process_mask(mask, threshold=0.5, min_size=30):
    """
    Post-process a predicted mask by thresholding and removing small regions
    
    Args:
        mask (numpy.ndarray): Predicted mask with values in [0, 1]
        threshold (float): Threshold for binarizing mask
        min_size (int): Minimum size of regions to keep
        
    Returns:
        numpy.ndarray: Binary mask with values 0 or 1
    """
    import cv2
    
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Initialize output mask
    output_mask = np.zeros_like(binary_mask)
    
    # Keep only components larger than min_size (skip label 0, which is background)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output_mask[labels == i] = 1
            
    return output_mask 