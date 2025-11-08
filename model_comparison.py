#!/usr/bin/env python3
"""
Model Comparison Analysis: resnet18_stl10.pth vs resnet18_linecolor.pth

This script compares two ResNet-18 models using Representational Similarity Analysis (RSA):
1. Whether the manifold structure is preserved between models
2. If there's more of a translation vs structural change
3. RSA correlation analysis to determine if correlation matrices are 90%+ correlated

Models:
- resnet18_stl10.pth: Color-only trained model
- resnet18_linecolor.pth: Line→Color transfer learning model

Analysis:
- Uses 100 images per class (1000 total images)
- Creates correlation matrices (-1 to 1) using correlation coefficients
- Creates both full (1000x1000) and class-averaged (10x10) correlation matrices
- Calculates Pearson and Spearman correlations between matrices
- Visualizes results to understand manifold structure preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import pandas as pd
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

def create_class_averaged_correlation_matrix(features, labels):
    """Create 10x10 class-averaged correlation matrix (-1 to 1)"""
    n_classes = 10
    class_averaged_matrix = np.zeros((n_classes, n_classes))
    
    # Convert to numpy if needed
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    
    for i in range(n_classes):
        for j in range(n_classes):
            # Get indices for classes i and j
            class_i_indices = np.where(labels == i)[0]
            class_j_indices = np.where(labels == j)[0]
            
            # Calculate average correlation coefficient between all pairs of images from class i and class j
            correlations = []
            for idx_i in class_i_indices:
                for idx_j in class_j_indices:
                    # Calculate correlation coefficient
                    corr_coef = np.corrcoef(features[idx_i], features[idx_j])[0, 1]
                    # Handle NaN case (when features are identical)
                    if np.isnan(corr_coef):
                        corr_coef = 1.0  # Identical features have correlation 1
                    correlations.append(corr_coef)
            
            class_averaged_matrix[i, j] = np.mean(correlations)
    
    return class_averaged_matrix

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

def analyze_correlation_matrix_correlation(matrix1, matrix2, model1_name, model2_name):
    """Analyze correlation between two correlation matrices"""
    # Flatten upper triangular matrices (excluding diagonal)
    n = matrix1.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    
    matrix1_flat = matrix1[triu_indices]
    matrix2_flat = matrix2[triu_indices]
    
    # Calculate Pearson correlation (RSA standard)
    correlation, p_value = pearsonr(matrix1_flat, matrix2_flat)
    
    # Calculate mean absolute difference
    mean_abs_diff = np.mean(np.abs(matrix1_flat - matrix2_flat))
    
    # Calculate Spearman correlation (alternative RSA metric)
    from scipy.stats import spearmanr
    spearman_corr, spearman_p = spearmanr(matrix1_flat, matrix2_flat)
    
    # Calculate structural similarity for correlation matrices
    c1, c2 = 0.01, 0.03  # Constants for SSIM
    mu1, mu2 = np.mean(matrix1), np.mean(matrix2)
    sigma1, sigma2 = np.std(matrix1), np.std(matrix2)
    sigma12 = np.mean((matrix1 - mu1) * (matrix2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
    
    return {
        'pearson_correlation': correlation,
        'pearson_p_value': p_value,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'mean_abs_diff': mean_abs_diff,
        'structural_similarity': ssim,
        'is_90_percent_correlated': correlation >= 0.9
    }


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model paths
    model_paths = {
        'Color Only': '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_color.pth',
        'Line Only': '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_line.pth',
        'Line→Color': '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_linecolor.pth',
        'Color→Line': '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_colorline.pth',
        'Interleaved': '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_interleaved.pth'
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
        img_dir='/user_data/georgeli/workspace/STL10-Resnet-18/STL10/test_images',
        json_file='/user_data/georgeli/workspace/STL10-Resnet-18/STL10/test.json',
        transform=transform
    )
    
    # Load line images dataset
    line_dataset = STL10Dataset(
        img_dir='/user_data/georgeli/workspace/STL10-Resnet-18/STL10-Line/test_images',
        json_file='/user_data/georgeli/workspace/STL10-Resnet-18/STL10-Line/test.json',
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
    output_dir = '/user_data/georgeli/workspace/STL10-Resnet-18/manifold_alignment/model_comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
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
                same_domain_matrices, cross_domain_matrices, list(model_paths.keys()), layer_name, output_dir
            )
            
        except Exception as e:
            print(f"  Error analyzing layer {layer_name}: {str(e)}")
    
    # Save results
    print(f"\nSaving results...")
    
    # Save correlation summary
    summary_data = []
    for layer_name, layer_results in results.items():
        if layer_results is not None:
            corr_analysis = layer_results['correlation_analysis']
            summary_data.append({
                'layer': layer_name,
                'pearson_correlation': corr_analysis['pearson_correlation'],
                'spearman_correlation': corr_analysis['spearman_correlation'],
                'pearson_p_value': corr_analysis['pearson_p_value'],
                'spearman_p_value': corr_analysis['spearman_p_value'],
                'mean_abs_diff': corr_analysis['mean_abs_diff'],
                'structural_similarity': corr_analysis['structural_similarity'],
                'is_90_percent_correlated': corr_analysis['is_90_percent_correlated']
            })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(output_dir, 'correlation_summary.csv'), index=False)
    
    # Save detailed results as JSON
    def convert_numpy(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        return obj
    
    # Save only correlation analysis (not the full matrices)
    json_results = {}
    for layer_name, layer_results in results.items():
        if layer_results is not None:
            json_results[layer_name] = {
                'correlation_analysis': convert_numpy(layer_results['correlation_analysis']),
                'n_samples': len(layer_results['labels']),
                'feature_dim': layer_results['features1'].shape[1]
            }
    
    with open(os.path.join(output_dir, 'model_comparison_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print final summary
    print("\n" + "="*80)
    print("MODEL COMPARISON ANALYSIS RESULTS")
    print("="*80)
    print(f"Models compared: {list(model_paths.keys())}")
    print(f"Images analyzed: {len(selected_indices)} (100 per class)")
    print()
    
    print(f"{'Layer':<12} {'Pearson RSA':<12} {'Spearman RSA':<12} {'90%+ Corr':<12} {'Mean Diff':<12}")
    print("-" * 70)
    
    manifold_preserved_layers = 0
    total_layers = 0
    
    for layer_name, layer_results in results.items():
        if layer_results is not None:
            corr_analysis = layer_results['correlation_analysis']
            pearson_corr = corr_analysis['pearson_correlation']
            spearman_corr = corr_analysis['spearman_correlation']
            is_90_percent = corr_analysis['is_90_percent_correlated']
            mean_diff = corr_analysis['mean_abs_diff']
            
            print(f"{layer_name:<12} {pearson_corr:<12.3f} {spearman_corr:<12.3f} {str(is_90_percent):<12} {mean_diff:<12.3f}")
            
            total_layers += 1
            if is_90_percent:
                manifold_preserved_layers += 1
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    if total_layers > 0:
        preservation_rate = manifold_preserved_layers / total_layers
        print(f"Manifold structure preservation rate: {preservation_rate:.1%} ({manifold_preserved_layers}/{total_layers} layers)")
        
        if preservation_rate >= 0.75:
            print("CONCLUSION: Manifold structure is WELL PRESERVED between models")
            print("The models learn similar structural representations, suggesting")
            print("the transfer learning maintains the underlying manifold geometry.")
        elif preservation_rate >= 0.5:
            print("CONCLUSION: Manifold structure is PARTIALLY PRESERVED")
            print("Some layers maintain structure while others show more translation.")
        else:
            print("CONCLUSION: Manifold structure is NOT WELL PRESERVED")
            print("The models learn significantly different representations,")
            print("suggesting more of a translation than structural preservation.")
    
    print(f"\nResults saved to: {output_dir}")
    print("Visualizations created:")
    for layer_name in layer_names:
        print(f"  - domain_comparison_matrices_{layer_name}.png (2x5 grid: Same Domain vs Cross Domain for all 5 models)")

if __name__ == "__main__":
    main()
