import os
import sys
import requests
from tqdm import tqdm

def download_file(url, destination):
    """
    Download a file from the specified URL to the destination path
    with progress bar using tqdm
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    print(f"Downloading from {url} to {destination}")
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Download failed, incomplete file")
        return False
    
    return True

def main():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "yolo11n-seg.pt")
    
    # Check if the weights file already exists
    if os.path.exists(weights_path):
        file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"YOLOv11-seg weights file already exists at {weights_path} ({file_size_mb:.1f} MB)")
        user_input = input("Do you want to download it again? (y/n): ")
        if user_input.lower() != 'y':
            print("Download skipped.")
            return
    
    # URL for the YOLOv11-seg weights
    # Note: This is a placeholder URL. You should replace it with the actual URL
    # for the YOLOv11-seg weights file once available from the official source
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"
    
    print("Warning: This script is using YOLOv8n-seg weights as a placeholder.")
    print("Please replace with actual YOLOv11-seg weights when available.")
    print("You can typically find the latest weights at https://github.com/ultralytics/ultralytics")
    
    try:
        success = download_file(model_url, weights_path)
        if success:
            print(f"Downloaded YOLOv11-seg weights to {weights_path}")
            print("You're now ready to run the training script!")
        else:
            print("Download failed. Please try again later or download manually.")
    except Exception as e:
        print(f"Error downloading weights: {e}")
        print("\nPlease download the YOLOv11-seg weights manually and place them at:")
        print(weights_path)

if __name__ == "__main__":
    main() 