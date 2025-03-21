# AI Manipulation Detection via Segmentation

This project detects and segments regions in images that have been manipulated by AI using a deep learning segmentation model.

## Project Overview

The goal is to create a model that can accurately detect which regions of an image have been manipulated by AI. The model outputs a binary mask where:
- `0` = Not manipulated (real)
- `1` = AI-generated or edited (fake)

## Dataset

The dataset consists of:
- Training set: 15,000 manipulated images with corresponding binary masks
- Test set: 18,735 manipulated test images (without ground truth)

All images have a resolution of 256x256 pixels.

## Project Structure

```
ai-manipulation-detector/
│
├── data/                          # Dataset links or symbolic links
│
├── scripts/
│   ├── train.py                  # Training pipeline
│   ├── infer.py                  # Inference & RLE encoding
│   ├── utils.py                  # Helper functions (RLE encoder, mask postprocessing)
│   ├── dataset.py                # Dataset class for loading and preprocessing data
│   ├── models/                   # Model definitions
│   │   ├── unet.py               # U-Net implementation
│   │   └── deeplabv3.py          # DeepLabV3+ implementation (optional)
│
├── models/
│   └── model_weights.pth         # Saved model weights
│
├── outputs/
│   ├── predicted_masks/          # Binary mask outputs from inference
│   └── submission.csv            # Final CSV for submission
│
├── requirements.txt               # Required packages
└── README.md                      # Project documentation
```

## Model

We implement a U-Net segmentation model, which has proved effective for similar pixel-wise segmentation tasks.

## Training Pipeline

1. Load and preprocess training data
2. Data augmentation (rotation, flips, etc.)
3. Train U-Net model using BCE + Dice Loss
4. Validate using Dice coefficient as the evaluation metric
5. Save best model weights

## Inference Pipeline

1. Load trained model
2. Preprocess test images
3. Generate binary masks
4. Post-process masks (optional)
5. Convert masks to RLE format
6. Create submission CSV

## Usage

1. **Setup Environment**:
   ```
   pip install -r requirements.txt
   ```

2. **Training**:
   ```
   python scripts/train.py
   ```

3. **Inference**:
   ```
   python scripts/infer.py
   ```

This will generate a `submission.csv` file in the `outputs/` directory.

## Evaluation

The model is evaluated using the Dice coefficient between the predicted masks and ground truth masks. 