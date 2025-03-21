# YOLOv11 Segmentation Training

This repository contains scripts for training YOLOv11 segmentation models on custom datasets.

## Dataset Structure

The expected dataset structure is as follows:

```
train/
├── train/
│   ├── images/       # Training images
│   ├── masks/        # Binary segmentation masks
│   └── originals/    # Optional: original unmodified images
├── val/              # Optional: will be created by the prepare_data.py script
│   ├── images/       # Validation images
│   └── masks/        # Validation masks
```

## Binary Masks Requirements

- Masks should be binary images with the same dimensions as their corresponding images
- Masks should contain only two values:
  - 0 for background
  - 1 or 255 for the foreground/object
- Each mask file should have the same filename as its corresponding image

## Setup Instructions

1. Install the required dependencies:
   ```
   pip install ultralytics pillow pyyaml numpy
   ```

2. Download the YOLOv11 segmentation model weights:
   ```
   wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n-seg.pt
   ```

3. Prepare your dataset folder structure as described above

## Data Preparation

The `prepare_data.py` script helps you prepare your dataset for training:

```
python prepare_data.py --images train/train/images --masks train/train/masks
```

Optional arguments:
- `--create-val`: Create a validation split from training data
- `--val-ratio 0.2`: Validation split ratio (default: 0.2)
- `--convert-masks`: Automatically convert 0/1 masks to 0/255 format if needed

Example with all options:
```
python prepare_data.py --images train/train/images --masks train/train/masks --create-val --val-ratio 0.2 --convert-masks
```

## Training

To train the YOLOv11 segmentation model on your dataset:

```
python train_yolov11_seg.py
```

This will:
1. Create a dataset configuration YAML file
2. Load the YOLOv11-seg model
3. Train the model with the specified parameters

## Training Parameters

The default training parameters in `train_yolov11_seg.py` are suitable for most cases, but you may want to modify:

- `epochs`: Number of training epochs (default: 100)
- `imgsz`: Input image size (default: 640)
- `batch`: Batch size (default: 16)
- `device`: GPU device ID (default: '0', use '' for CPU)

You can also tweak augmentation parameters, learning rates, and other training hyperparameters in the script.

## Training Output

Training results will be saved to `runs/segment/yolov11_seg_train/` including:
- Best model weights
- Training plots
- Inference examples
- Results CSV

## Dataset Considerations

1. **Image-Mask Pairs**: Ensure each image has a corresponding mask with the same filename
2. **Binary Masks**: Masks should be binary (containing only 0s and 1s/255s)
3. **Class Names**: Update the `names` field in the dataset.yaml file if you have multiple classes
4. **Validation Split**: Create a validation split for more reliable model evaluation

## Troubleshooting

- **Memory Issues**: Reduce batch size or image size
- **CUDA Out of Memory**: Use a smaller model variant or reduce batch size
- **Poor Results**: Increase training epochs, check mask quality, or adjust learning rate

Good luck with your YOLOv11 segmentation training! 