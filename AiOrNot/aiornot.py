import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import argparse

def classify_image(image_path):
    """
    Classify an image as either 'Real' or 'AI' using the Hugging Face model.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Classification result ('Real' or 'AI')
        float: Confidence score
    """
    # Load the image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None
    
    # Load the model and feature extractor
    labels = ["Real", "AI"]
    feature_extractor = AutoFeatureExtractor.from_pretrained("Nahrawy/AIorNot")
    model = AutoModelForImageClassification.from_pretrained("Nahrawy/AIorNot")
    
    # Prepare input and get prediction
    inputs = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get prediction and confidence
    prediction = logits.argmax(-1).item()
    label = labels[prediction]
    
    # Calculate confidence (using softmax to get probabilities)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probs[0][prediction].item() * 100
    
    return label, confidence

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Detect if an image is AI-generated or real')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()
    
    # Classify the image
    result, confidence = classify_image(args.image_path)
    
    if result:
        print(f"Classification: {result}")
        print(f"Confidence: {confidence:.2f}%") 