import os
import argparse
from ultralytics import YOLO

def train_simple(epochs=10, batch=8, yaml_path=None, use_gpu=True, img_size=256):
    """Simple training function optimized for 256x256 AI manipulation detection"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for YAML configuration
    if yaml_path is None:
        yaml_path = os.path.join(script_dir, 'simple_dataset.yaml')
    
    if not os.path.exists(yaml_path):
        print(f"YAML configuration not found at {yaml_path}")
        print("Please run fix_mask_format.py first to create the YAML file")
        return
    
    # Check for YOLO model weights
    model_paths = [
        os.path.join(script_dir, "yolo11n-seg.pt"),        # YOLOv11 in script dir
        os.path.join(script_dir, "yolov8n-seg.pt"),        # YOLOv8 in script dir
        "yolo11n-seg.pt",                                 # YOLOv11 in current dir
        "yolov8n-seg.pt"                                  # YOLOv8 in current dir
    ]
    
    model_file = None
    for path in model_paths:
        if os.path.exists(path):
            model_file = path
            print(f"Using YOLO model: {path}")
            break
    
    if model_file is None:
        print("No YOLO model weights found. Attempting to download yolov8n-seg.pt...")
        model_file = "yolov8n-seg.pt"  # Use the name and let YOLO download it
    
    # Set device based on GPU availability and user preference
    device = '0' if use_gpu else 'cpu'
    
    try:
        print(f"Loading model: {model_file}")
        model = YOLO(model_file)
        
        print("\n=== Starting YOLO Training for AI Manipulation Detection ===")
        print(f"Using dataset config: {yaml_path}")
        print(f"Epochs: {epochs}, Batch size: {batch}")
        print(f"Image size: {img_size}x{img_size} (matches your dataset resolution)")
        print(f"Device: {'GPU (CUDA)' if use_gpu else 'CPU'}")
        
        # Simple training with minimal parameters optimized for this task
        model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch,
            device=device,        # Use GPU if available
            exist_ok=True,
            verbose=True,
            patience=30,          # Early stopping if no improvement for 30 epochs
            cos_lr=True,          # Cosine learning rate schedule
            close_mosaic=10,      # Disable mosaic augmentation in final epochs
            degrees=0.0,          # Less rotation for rectangular objects
            scale=0.5,            # Scale augmentation
            hsv_h=0.015,          # HSV augmentation
            hsv_s=0.7,
            hsv_v=0.4,
            single_cls=True       # Single class (manipulated vs real)
        )
        
        print("\nTraining completed successfully!")
        print("\nTo predict on new images, use:")
        print("  from ultralytics import YOLO")
        print("  model = YOLO('runs/segment/train/weights/best.pt')")
        print("  results = model.predict('path/to/image.jpg', save=True)")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        
        if "CUDA out of memory" in str(e):
            print("\nGPU memory error detected. Try the following:")
            print("1. Reduce batch size: --batch 4 or --batch 2")
            print("2. Try running on CPU: --no-gpu")
        else:
            print("\nTroubleshooting tips:")
            print("1. Make sure your masks are properly formatted (run fix_mask_format.py)")
            print("2. Check that your YAML configuration is correct")
            print("3. Try with a smaller batch size (--batch 4)")
            print("4. If GPU issues persist, try CPU mode: --no-gpu")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Manipulation Detection Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (reduce if GPU memory issues)")
    parser.add_argument("--data", type=str, default=None, help="Path to YAML dataset configuration")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU training even if GPU is available")
    parser.add_argument("--img-size", type=int, default=256, help="Image size for training (default: 256)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    
    # Run training
    train_simple(
        epochs=args.epochs, 
        batch=args.batch, 
        yaml_path=args.data,
        use_gpu=not args.no_gpu,
        img_size=args.img_size
    )

if __name__ == "__main__":
    main() 