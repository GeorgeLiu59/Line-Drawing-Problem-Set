import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from tqdm import tqdm

# Configuration - just put your model paths here
MODEL_PATHS = [
    '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_stl10.pth',
    '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_stl10_line.pth',
    '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_colorline.pth',
    '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_linecolor.pth',
    '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_interleaved.pth'
]

# Photograph Data
# TEST_JSON_PATH = '/lab_data/leelab/georgeli/datasets/STL10/test.json'
# TEST_IMAGES_DIR = '/lab_data/leelab/georgeli/datasets/STL10'

# Line Drawing Data
TEST_JSON_PATH = '/user_data/georgeli/workspace/STL10-Resnet-18/STL10-Line/test.json'
TEST_IMAGES_DIR = '/user_data/georgeli/workspace/STL10-Resnet-18/STL10-Line'


def create_model():
    """Create a fresh ResNet18 model"""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Linear(512, 10))
    return model

def load_model(model_path, device):
    """Load a ResNet18 model from the given path"""
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def evaluate_models():
    """Evaluate the specified models on the test set"""
    print("Model Accuracy Evaluation")
    print("=" * 50)
    
    # Define the transformation pipeline
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load models
    models_dict = {}
    print("Loading models...")
    for i, model_path in enumerate(MODEL_PATHS):
        if os.path.exists(model_path):
            try:
                model = load_model(model_path, device)
                model_name = f"Model_{i+1}_{os.path.basename(model_path)}"
                models_dict[model_name] = {
                    'model': model,
                    'correct': 0,
                    'total': 0
                }
                print(f"✓ Loaded: {model_name}")
            except Exception as e:
                print(f"✗ Error loading {model_path}: {e}")
        else:
            print(f"✗ Model not found: {model_path}")
    
    if len(models_dict) == 0:
        print("No models loaded successfully!")
        return
    
    # Load test data
    print(f"\nLoading test data from {TEST_JSON_PATH}")
    with open(TEST_JSON_PATH, 'r') as f:
        test_data = json.load(f)
    print(f"Test set size: {len(test_data)} images")
    
    # Evaluate models
    print("\nEvaluating models...")
    with torch.no_grad():
        for item in tqdm(test_data, desc="Processing images"):
            image_path = os.path.join(TEST_IMAGES_DIR, item['file'])
            label = item['label']
            
            if not os.path.exists(image_path):
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = test_transform(image).unsqueeze(0).to(device)
                
                # Test on all models
                for model_name, model_info in models_dict.items():
                    outputs = model_info['model'](image_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    model_info['total'] += 1
                    model_info['correct'] += (predicted == label).sum().item()
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
    
    # Print results
    print("\nResults:")
    print("=" * 50)
    for model_name, model_info in models_dict.items():
        if model_info['total'] > 0:
            accuracy = model_info['correct'] / model_info['total'] * 100
            print(f"{model_name}: {accuracy:.2f}% ({model_info['correct']}/{model_info['total']})")
        else:
            print(f"{model_name}: No valid predictions")

if __name__ == "__main__":
    evaluate_models()