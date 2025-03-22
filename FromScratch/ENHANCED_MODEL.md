# Enhanced UNet Model for AI Manipulation Detection

This document explains the enhanced UNet model implementation and how to use it.

## Model Improvements

The enhanced UNet model (`models/unet_enhanced.py`) includes several improvements over the standard UNet:

1. **Residual Connections**: Each convolutional block now includes residual connections, which help with gradient flow during training and can lead to better convergence.

2. **Attention Mechanisms**: Two types of attention are included:
   - Channel Attention: Helps the model focus on the most relevant feature channels
   - Spatial Attention: Helps the model focus on the most important spatial regions

3. **Dropout Regularization**: Dropout layers are added throughout the model to prevent overfitting, especially important when training with limited data.

4. **Better Optimization**: The enhanced training script includes:
   - Learning rate schedulers (Cosine Annealing, Step, or ReduceLROnPlateau)
   - AdamW optimizer with weight decay
   - More detailed progress reporting

## How to Use the Enhanced Model

### Training

To train the model with enhanced architecture and training techniques:

```bash
python3 run.py --create-val --train --enhanced --epochs 30
```

Additional parameters you can customize:

```bash
python3 run.py --create-val --train --enhanced \
    --epochs 30 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --dropout 0.1 \
    --optimizer adamw \  # Options: adam, adamw
    --scheduler cosine \ # Options: cosine, step, reduce, none
    --weight-decay 1e-5
```

### Inference

Inference works the same way with the enhanced model - just use the inference option after training:

```bash
python3 run.py --inference
```

### Expected Improvements

The enhanced model should provide several benefits:

1. **Better Accuracy**: The attention mechanisms help the model focus on important features, leading to more accurate predictions.

2. **Faster Convergence**: Residual connections and better optimizers typically result in faster and more stable convergence.

3. **Better Generalization**: Dropout and regularization techniques help the model generalize better to unseen data.

4. **More Robust Training**: Learning rate schedulers help the model navigate difficult loss landscapes.

## TensorBoard Visualization

You can monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

This will show you learning curves, losses, and evaluation metrics during training.

## Comparing with Base Model

To compare performance with the base UNet model, you can train both models and compare their validation Dice scores:

1. Train base model:
```bash
python3 run.py --create-val --train --epochs 30
```

2. Train enhanced model:
```bash
python3 run.py --create-val --train --enhanced --epochs 30
```

3. Compare the final validation Dice scores and the quality of predictions.

## Troubleshooting

If you encounter any issues:

1. **Memory Errors**: Try reducing batch size
2. **Slow Training**: Reduce the number of workers
3. **Overfitting**: Increase dropout or weight decay
4. **Underfitting**: Decrease dropout, increase learning rate or number of epochs 