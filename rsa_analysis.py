#!/usr/bin/env python3
"""
Manifold Alignment Analysis for ResNet18 Models

This script implements:
1. Cosine similarity between line and color representations
2. Representational Similarity Analysis (RSA) using RDM
3. Analysis across different training strategies and network layers

Models Analyzed:
- Color Only (resnet18_color.pth)
- Line Only (resnet18_line.pth) 
- Line → Color (resnet18_linecolor.pth)
- Color → Line (resnet18_colorline.pth)
- Interleaved (resnet18_interleaved.pth)
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
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

def extract_corresponding_features(model, line_dataset, color_dataset, layer_name, device, max_samples=10):
    feature_extractor = FeatureExtractor(model, layer_name)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    line_features = []
    color_features = []
    labels = []
    filenames = []
    
    color_filename_to_idx = {}
    for idx, annotation in enumerate(color_dataset.annotations):
        filename = os.path.basename(annotation['file'])
        color_filename_to_idx[filename] = idx
    
    with torch.no_grad():
        for i in tqdm(range(min(max_samples, len(line_dataset))), desc=f"Extracting corresponding {layer_name}"):
            line_image, line_label, line_filename = line_dataset[i]
            line_filename_base = os.path.basename(line_filename)
            
            if line_filename_base in color_filename_to_idx:
                color_idx = color_filename_to_idx[line_filename_base]
                color_image, color_label, color_filename = color_dataset[color_idx]
                
                if line_label == color_label:
                    # Extract line features first
                    line_batch = line_image.unsqueeze(0).to(device)
                    _ = feature_extractor(line_batch)
                    # Immediately clone and process to avoid any state issues
                    line_layer_features = feature_extractor.features[layer_name].clone()
                    line_layer_features = line_layer_features.view(line_layer_features.size(0), -1)
                    line_feat_np = line_layer_features.cpu().numpy()
                    del line_layer_features
                    
                    # Extract color features - this will overwrite feature_extractor.features[layer_name]
                    color_batch = color_image.unsqueeze(0).to(device)
                    _ = feature_extractor(color_batch)
                    # Immediately clone and process
                    color_layer_features = feature_extractor.features[layer_name].clone()
                    color_layer_features = color_layer_features.view(color_layer_features.size(0), -1)
                    color_feat_np = color_layer_features.cpu().numpy()
                    del color_layer_features
                    
                    # Ensure proper shapes for stacking
                    if len(line_feat_np.shape) == 1:
                        line_feat_np = line_feat_np.reshape(1, -1)
                    if len(color_feat_np.shape) == 1:
                        color_feat_np = color_feat_np.reshape(1, -1)
                    
                    line_features.append(line_feat_np)
                    color_features.append(color_feat_np)
                    labels.append(line_label)
                    filenames.append(line_filename_base)
                    
                    torch.cuda.empty_cache()
    
    return (np.vstack(line_features), np.vstack(color_features), 
            np.array(labels), filenames)

def cosine_similarity_analysis(line_features, color_features):
    """Compute cosine similarity between corresponding line and color features"""
    if isinstance(line_features, torch.Tensor):
        line_features = line_features.numpy()
    if isinstance(color_features, torch.Tensor):
        color_features = color_features.numpy()
    
    def cosine_similarity_pair(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        return np.dot(a_norm, b_norm)
    
    corresponding_similarities = []
    for i in range(len(line_features)):
        sim = cosine_similarity_pair(line_features[i], color_features[i])
        corresponding_similarities.append(sim)
    
    mean_corresponding_similarity = np.mean(corresponding_similarities)
    
    return {
        'mean_corresponding_similarity': mean_corresponding_similarity
    }

def representational_similarity_analysis(line_features, color_features):
    """
    Compute Representational Similarity Analysis (RSA) using RDM.
    
    Args:
        line_features: numpy array of shape (n_samples, n_features) for line images
        color_features: numpy array of shape (n_samples, n_features) for color images
    
    Returns:
        dict with 'rsa_correlation' and 'rsa_p_value'
    """
    # Ensure features are numpy arrays and properly shaped
    if isinstance(line_features, torch.Tensor):
        line_features = line_features.cpu().numpy()
    if isinstance(color_features, torch.Tensor):
        color_features = color_features.cpu().numpy()
    
    # Ensure 2D arrays (samples x features)
    if len(line_features.shape) == 1:
        line_features = line_features.reshape(1, -1)
    if len(color_features.shape) == 1:
        color_features = color_features.reshape(1, -1)
    
    # Check for NaN or infinite values
    if np.any(np.isnan(line_features)) or np.any(np.isinf(line_features)):
        raise ValueError("Line features contain NaN or infinite values")
    if np.any(np.isnan(color_features)) or np.any(np.isinf(color_features)):
        raise ValueError("Color features contain NaN or infinite values")
    
    # Ensure we have the same number of samples
    if len(line_features) != len(color_features):
        raise ValueError(f"Mismatch in number of samples: {len(line_features)} vs {len(color_features)}")
    
    n_samples = len(line_features)
    
    # Compute RDM (Representational Dissimilarity Matrix)
    def compute_rdm(features):
        """Compute Representational Dissimilarity Matrix"""
        n = len(features)
        rdm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(features[i] - features[j])
                    if np.isnan(dist) or np.isinf(dist):
                        raise ValueError(f"Invalid distance computed between samples {i} and {j}")
                    rdm[i, j] = dist
        return rdm
    
    line_rdm = compute_rdm(line_features)
    color_rdm = compute_rdm(color_features)
    
    # Extract upper triangular parts (excluding diagonal) - standard RSA approach
    triu_indices = np.triu_indices(n_samples, k=1)
    line_distances = line_rdm[triu_indices]
    color_distances = color_rdm[triu_indices]
    
    # Check for constant distance vectors (would make correlation undefined)
    line_var = np.var(line_distances)
    color_var = np.var(color_distances)
    
    if line_var == 0 or color_var == 0:
        # If all distances are the same, correlation is undefined, return 0
        return {
            'rsa_correlation': 0.0,
            'rsa_p_value': 1.0
        }
    
    # Use Pearson correlation as the standard RSA metric
    correlation, p_value = pearsonr(line_distances, color_distances)
    
    # Check for invalid correlation values
    if np.isnan(correlation) or np.isinf(correlation):
        raise ValueError(f"Invalid correlation computed: {correlation}")
    
    return {
        'rsa_correlation': correlation,
        'rsa_p_value': p_value
    }

def create_visualizations(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(results.keys())
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    
    # Cosine Similarity Comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, layer in enumerate(layers):
        ax = axes[i]
        similarities = []
        model_names = []
        
        for model in models:
            if model in results and layer in results[model] and results[model][layer] is not None:
                sim = results[model][layer]['cosine_similarity']['mean_corresponding_similarity']
                similarities.append(sim)
                model_names.append(model)
        
        if similarities:
            bars = ax.bar(model_names, similarities, alpha=0.7)
            ax.set_title(f'Cosine Similarity - {layer}')
            ax.set_ylabel('Cosine Similarity')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, sim in zip(bars, similarities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{sim:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    cosine_data = []
    for model in models:
        row = []
        for layer in layers:
            if model in results and layer in results[model] and results[model][layer] is not None:
                sim = results[model][layer]['cosine_similarity']['mean_corresponding_similarity']
                row.append(sim)
            else:
                row.append(np.nan)
        cosine_data.append(row)
    
    sns.heatmap(cosine_data, xticklabels=layers, yticklabels=models, 
                annot=True, fmt='.3f', cmap='viridis', ax=ax)
    ax.set_title('Cosine Similarity Heatmap')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results as JSON
    with open(os.path.join(output_dir, 'manifold_alignment_results.json'), 'w') as f:
        def convert_numpy(obj):
            """Recursively convert numpy types to Python native types for JSON serialization"""
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
        
        # Convert all results to JSON-serializable format
        json_results = convert_numpy(results)
        
        json.dump(json_results, f, indent=2)
    
    # Save summary table as CSV
    summary_data = []
    for model_name, model_results in results.items():
        for layer_name, layer_results in model_results.items():
            if layer_results is not None:
                summary_data.append({
                    'model': model_name,
                    'layer': layer_name,
                    'cosine_similarity': layer_results['cosine_similarity']['mean_corresponding_similarity'],
                    'rsa_correlation': layer_results['rsa']['rsa_correlation'],
                    'n_samples': layer_results['n_samples']
                })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(output_dir, 'manifold_alignment_summary.csv'), index=False)
    
    print(f"Results saved to {output_dir}")

def main():
    # Configuration - update paths to match your setup
    MODELS_DIR = 'Models'
    LINE_IMAGES_DIR = 'STL10-Line/test_images'
    LINE_JSON_FILE = 'STL10-Line/test.json'
    COLOR_IMAGES_DIR = 'STL10/test_images'
    COLOR_JSON_FILE = 'STL10/test.json'
    OUTPUT_DIR = 'results'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_paths = {
        'Color Only': os.path.join(MODELS_DIR, 'resnet18_color.pth'),
        'Line Only': os.path.join(MODELS_DIR, 'resnet18_line.pth'),
        'Line → Color': os.path.join(MODELS_DIR, 'resnet18_linecolor.pth'),
        'Color → Line': os.path.join(MODELS_DIR, 'resnet18_colorline.pth'),
        'Interleaved': os.path.join(MODELS_DIR, 'resnet18_interleaved.pth')
    }
    
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    
    transform = get_transform()
    
    line_dataset = STL10Dataset(
        img_dir=LINE_IMAGES_DIR,
        json_file=LINE_JSON_FILE,
        transform=transform
    )
    
    color_dataset = STL10Dataset(
        img_dir=COLOR_IMAGES_DIR,
        json_file=COLOR_JSON_FILE,
        transform=transform
    )
    
    results = {}
    
    print("Starting manifold alignment analysis...")
    
    for model_name, model_path in model_paths.items():
        print(f"\nAnalyzing model: {model_name}")
        print("=" * 50)
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        model = load_model(model_path, device)
        results[model_name] = {}
        
        for layer_name in layer_names:
            print(f"\nAnalyzing layer: {layer_name}")
            
            try:
                line_features, color_features, labels, filenames = extract_corresponding_features(
                    model, line_dataset, color_dataset, layer_name, device, max_samples=20
                )
                
                cosine_results = cosine_similarity_analysis(line_features, color_features)
                rsa_results = representational_similarity_analysis(line_features, color_features)
                
                results[model_name][layer_name] = {
                    'cosine_similarity': cosine_results,
                    'rsa': rsa_results,
                    'n_samples': len(line_features)
                }
                
                print(f"  Cosine similarity: {cosine_results['mean_corresponding_similarity']:.4f}")
                print(f"  RSA correlation: {rsa_results['rsa_correlation']:.4f}")
                print(f"  Samples: {len(line_features)}")
                
            except Exception as e:
                print(f"  Error analyzing layer {layer_name}: {str(e)}")
                results[model_name][layer_name] = None
    
    save_results(results, OUTPUT_DIR)
    
    print("\nCreating visualizations...")
    create_visualizations(results, OUTPUT_DIR)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for model_name, model_results in results.items():
        if model_results:
            print(f"\n{model_name.upper()}")
            print("-" * 40)
        print(f"{'Layer':<12} {'Cosine Sim':<12} {'RSA Corr':<12}")
        print("-" * 40)
        
        for layer_name, layer_results in model_results.items():
            if layer_results is not None:
                sim = layer_results['cosine_similarity']['mean_corresponding_similarity']
                rsa_corr = layer_results['rsa']['rsa_correlation']
                print(f"{layer_name:<12} {sim:<12.3f} {rsa_corr:<12.3f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
