#!/usr/bin/env python3
"""
Model Comparison Analysis for ResNet-18 Models

This script compares ResNet-18 models using correlation matrices:
- Creates same-domain and cross-domain correlation matrices
- Visualizes correlation patterns across different training strategies

Models analyzed:
- Color Only, Line Only, Line→Color, Color→Line, Interleaved

Analysis:
- Uses 100 images per class (1000 total images)
- Creates correlation matrices (-1 to 1) using correlation coefficients
- Visualizes same-domain vs cross-domain correlation patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class STL10Dataset(Dataset):
    """Custom dataset for STL10 images"""
    def __init__(self, img_dir, json_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(json_file, 'r') as f:
            self.annotations = json.load(f)
            
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, os.path.basename(self.annotations[idx]['file']))
        image = Image.open(img_path).convert('RGB')
        label = self.annotations[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label, self.annotations[idx]['file']

class FeatureExtractor(nn.Module):
    """Feature extractor for different ResNet18 layers"""
    def __init__(self, model, layer_name):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to extract features from specific layers"""
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook
        
        layer_mapping = {
            'conv1': self.model.conv1,
            'bn1': self.model.bn1,
            'relu': self.model.relu,
            'maxpool': self.model.maxpool,
            'layer1': self.model.layer1,
            'layer2': self.model.layer2,
            'layer3': self.model.layer3,
            'layer4': self.model.layer4,
            'avgpool': self.model.avgpool,
            'fc': self.model.fc
        }
        
        if self.layer_name in layer_mapping:
            layer_mapping[self.layer_name].register_forward_hook(get_activation(self.layer_name))
    
    def forward(self, x):
        _ = self.model(x)
        return self.features[self.layer_name]

def create_model():
    """Create ResNet18 model"""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Linear(512, 10))
    return model

def load_model(model_path, device):
    """Load trained model"""
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_transform():
    """Get image transformation"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def select_images_per_class(dataset, images_per_class=None):
    """Select images from each class. If images_per_class is None, use all available images."""
    selected_indices = []
    selected_info = []
    
    # First pass: collect all images by class
    class_images = {}
    for idx, annotation in enumerate(dataset.annotations):
        label = annotation['label']
        if label not in class_images:
            class_images[label] = []
        class_images[label].append({
            'idx': idx,
            'label': label,
            'filename': os.path.basename(annotation['file'])
        })
    
    # Second pass: select images from each class
    for class_label in range(10):  # Classes 0-9
        if class_label in class_images:
            if images_per_class is None:
                # Use all images from this class
                selected_class_images = class_images[class_label]
            else:
                # Use only the specified number of images
                selected_class_images = class_images[class_label][:images_per_class]
            
            for img_info in selected_class_images:
                selected_indices.append(img_info['idx'])
                selected_info.append(img_info)
        else:
            print(f"Warning: No images found for class {class_label}")
    
    print(f"Selected {len(selected_indices)} images:")
    for label in range(10):
        count = len([info for info in selected_info if info['label'] == label])
        print(f"  Class {label}: {count} images")
    
    return selected_indices, selected_info

def extract_features_for_selected(model, dataset, selected_indices, layer_name, device):
    """Extract features for selected images"""
    feature_extractor = FeatureExtractor(model, layer_name)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    features = []
    labels = []
    filenames = []
    
    with torch.no_grad():
        for idx in tqdm(selected_indices, desc=f"Extracting {layer_name} features"):
            image, label, filename = dataset[idx]
            image_batch = image.unsqueeze(0).to(device)
            
            _ = feature_extractor(image_batch)
            layer_features = feature_extractor.features[layer_name]
            
            # Handle different layer output shapes
            if len(layer_features.shape) > 2:
                layer_features = F.adaptive_avg_pool2d(layer_features, (1, 1))
            layer_features = layer_features.view(layer_features.size(0), -1)
            
            features.append(layer_features.cpu().numpy())
            labels.append(label)
            filenames.append(os.path.basename(filename))
    
    return np.vstack(features), np.array(labels), filenames

def create_representational_similarity_matrix(features):
    """Create Representational Similarity Matrix using correlation coefficients (-1 to 1)"""
    # Convert to numpy if needed
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    
    # Calculate pairwise correlation coefficients
    correlation_matrix = np.corrcoef(features)
    
    return correlation_matrix

def create_cross_domain_correlation_matrix(color_features, line_features):
    """Create cross-domain correlation matrix (Color vs Line)"""
    # Convert to numpy if needed
    if isinstance(color_features, torch.Tensor):
        color_features = color_features.numpy()
    if isinstance(line_features, torch.Tensor):
        line_features = line_features.numpy()
    
    # Calculate correlation between color and line features
    # Each row is a color image, each column is a line image
    cross_domain_matrix = np.corrcoef(color_features, line_features)
    
    # Extract only the cross-domain part (color vs line)
    n_color = color_features.shape[0]
    n_line = line_features.shape[0]
    cross_domain_matrix = cross_domain_matrix[:n_color, n_color:]
    
    return cross_domain_matrix

def create_domain_comparison_visualization(same_domain_matrices, cross_domain_matrices, model_names, layer_name, output_dir):
    """Create 2x5 grid showing Same Domain vs Cross Domain correlation matrices for all 5 models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create 2x5 grid for all 5 models
    fig, axes = plt.subplots(2, 5, figsize=[_/2.54 for _ in [25, 10]], dpi=300)
    
    # Add overall title with layer name
    fig.suptitle(f'Domain Comparison Matrices - {layer_name}', size=14, fontweight='bold', y=0.95)
    
    for i, (same_domain_matrix, cross_domain_matrix, model_name) in enumerate(zip(same_domain_matrices, cross_domain_matrices, model_names)):
        # Top row: Same Domain matrices (Color vs Color, Line vs Line)
        im1 = axes[0, i].imshow(same_domain_matrix, vmax=1, vmin=-1, cmap='bwr')
        axes[0, i].set_title(f'{model_name}\n(Same Domain)', size=8, pad=3, fontweight='bold')
        axes[0, i].set_xlabel('Images', size=7, labelpad=1)
        axes[0, i].set_ylabel('Images', size=7, labelpad=1)
        axes[0, i].tick_params('both', labelsize=5, size=2, pad=1)
        
        # Bottom row: Cross Domain matrices (Color vs Line)
        im2 = axes[1, i].imshow(cross_domain_matrix, vmax=1, vmin=-1, cmap='bwr')
        axes[1, i].set_title(f'{model_name}\n(Color vs Line)', size=8, pad=3, fontweight='bold')
        axes[1, i].set_xlabel('Images', size=7, labelpad=1)
        axes[1, i].set_ylabel('Images', size=7, labelpad=1)
        axes[1, i].tick_params('both', labelsize=5, size=2, pad=1)
        
        # Add colorbars
        cb1 = fig.colorbar(im1, ax=axes[0, i], shrink=0.8, pad=0.1)
        cb1.ax.tick_params(labelsize=5, size=2, pad=1)
        cb1.ax.set_ylabel('Correlation', size=7)
        
        cb2 = fig.colorbar(im2, ax=axes[1, i], shrink=0.8, pad=0.1)
        cb2.ax.tick_params(labelsize=5, size=2, pad=1)
        cb2.ax.set_ylabel('Correlation', size=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'domain_comparison_matrices_{layer_name}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run model comparison analysis"""
    # Configuration - update paths to match your setup
    MODELS_DIR = 'Models'
    LINE_IMAGES_DIR = 'STL10-Line/test_images'
    LINE_JSON_FILE = 'STL10-Line/test.json'
    COLOR_IMAGES_DIR = 'STL10/test_images'
    COLOR_JSON_FILE = 'STL10/test.json'
    OUTPUT_DIR = 'model_comparison_results'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model paths
    model_paths = {
        'Color Only': os.path.join(MODELS_DIR, 'resnet18_color.pth'),
        'Line Only': os.path.join(MODELS_DIR, 'resnet18_line.pth'),
        'Line→Color': os.path.join(MODELS_DIR, 'resnet18_linecolor.pth'),
        'Color→Line': os.path.join(MODELS_DIR, 'resnet18_colorline.pth'),
        'Interleaved': os.path.join(MODELS_DIR, 'resnet18_interleaved.pth')
    }
    
    # Check if models exist
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"Error: {model_name} model not found: {model_path}")
            return
    
    # Load datasets
    transform = get_transform()
    
    # Load color images dataset
    color_dataset = STL10Dataset(
        img_dir=COLOR_IMAGES_DIR,
        json_file=COLOR_JSON_FILE,
        transform=transform
    )
    
    # Load line images dataset
    line_dataset = STL10Dataset(
        img_dir=LINE_IMAGES_DIR,
        json_file=LINE_JSON_FILE,
        transform=transform
    )
    
    # Select 100 images per class (1000 total) from color dataset
    print("Selecting 100 images per class...")
    selected_indices, selected_info = select_images_per_class(color_dataset, images_per_class=100)
    
    # Load models
    print(f"\nLoading models...")
    models = {}
    for model_name, model_path in model_paths.items():
        print(f"  Loading {model_name}...")
        models[model_name] = load_model(model_path, device)
    
    # Define layers to analyze
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nStarting model comparison analysis...")
    print(f"Models: {list(model_paths.keys())}")
    print(f"Images: {len(selected_indices)} (100 per class)")
    print(f"Layers: {layer_names}")
    print("="*80)
    
    for layer_name in layer_names:
        print(f"\nAnalyzing layer: {layer_name}")
        print("-" * 50)
        
        try:
            # Extract features from all models using BOTH color and line images
            all_color_features = []
            all_line_features = []
            all_labels = []
            all_filenames = []
            
            for model_name, model in models.items():
                print(f"  Extracting features from {model_name}...")
                
                # Extract features from color images
                print(f"    Extracting color features...")
                color_features, color_labels, color_filenames = extract_features_for_selected(
                    model, color_dataset, selected_indices, layer_name, device
                )
                
                # Extract features from line images
                print(f"    Extracting line features...")
                line_features, line_labels, line_filenames = extract_features_for_selected(
                    model, line_dataset, selected_indices, layer_name, device
                )
                
                all_color_features.append(color_features)
                all_line_features.append(line_features)
                all_labels.append(color_labels)  # Use color labels as reference
                all_filenames.append(color_filenames)
            
            # Verify all models have the same labels
            for i in range(1, len(all_labels)):
                assert np.array_equal(all_labels[0], all_labels[i]), f"Labels don't match between models"
            
            # Create correlation matrices for all models
            print("  Creating correlation matrices for all models...")
            same_domain_matrices = []
            cross_domain_matrices = []
            
            for i, (color_features, line_features) in enumerate(zip(all_color_features, all_line_features)):
                model_name = list(model_paths.keys())[i]
                
                # Same domain: Use appropriate domain based on model training
                if model_name in ['Color Only', 'Line→Color', 'Interleaved']:
                    # These models are primarily color domain
                    same_domain_matrix = create_representational_similarity_matrix(color_features)
                else:  # ['Line Only', 'Color→Line']
                    # These models are primarily line domain
                    same_domain_matrix = create_representational_similarity_matrix(line_features)
                
                # Cross domain: Color vs Line correlation
                cross_domain_matrix = create_cross_domain_correlation_matrix(color_features, line_features)
                
                same_domain_matrices.append(same_domain_matrix)
                cross_domain_matrices.append(cross_domain_matrix)
            
            # Create 2x5 grid visualization: Same Domain vs Cross Domain for all 5 models
            print("  Creating 2x5 grid visualization (Same Domain vs Cross Domain)...")
            create_domain_comparison_visualization(
                same_domain_matrices, cross_domain_matrices, list(model_paths.keys()), layer_name, OUTPUT_DIR
            )
            
        except Exception as e:
            print(f"  Error analyzing layer {layer_name}: {str(e)}")
    
    print("\n" + "="*80)
    print("MODEL COMPARISON ANALYSIS COMPLETE")
    print("="*80)
    print(f"Models compared: {list(model_paths.keys())}")
    print(f"Images analyzed: {len(selected_indices)} (100 per class)")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("Visualizations created:")
    for layer_name in layer_names:
        print(f"  - domain_comparison_matrices_{layer_name}.png (2x5 grid: Same Domain vs Cross Domain for all 5 models)")

if __name__ == "__main__":
    main()
