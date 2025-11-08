#!/usr/bin/env python3
"""
Complete Manifold Alignment Analysis for ResNet18 Models

This script implements all the analysis methods requested:
1. Cosine similarity between line and color representations
2. Representational Similarity Analysis (RSA) 
3. Procrustes analysis for manifold alignment
4. Analysis across different training strategies and network layers

Research Questions Addressed:
- Does training with color first create a less structural prior?
- Does combined training enhance alignment but favor the line manifold?
- Can we establish this through latent representation analysis?

Models Analyzed:
- Color Only (resnet18_stl10.pth)
- Line Only (resnet18_stl10_line.pth) 
- Line → Color (resnet18_linecolor.pth)
- Color → Line (resnet18_colorline.pth)
- Combined (resnet18_learntodraw.pth)
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
from scipy.linalg import orthogonal_procrustes
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

def extract_features(model, dataloader, layer_name, device, max_samples=10):
    feature_extractor = FeatureExtractor(model, layer_name)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    features = []
    labels = []
    filenames = []
    
    with torch.no_grad():
        for i, (images, batch_labels, batch_filenames) in enumerate(tqdm(dataloader, desc=f"Extracting {layer_name}")):
            if len(features) >= max_samples:
                break
                
            images = images.to(device)
            _ = feature_extractor(images)
            
            layer_features = feature_extractor.features[layer_name]
            layer_features = layer_features.view(layer_features.size(0), -1)
            
            features.append(layer_features.cpu().numpy())
            labels.extend(batch_labels.numpy())
            filenames.extend(batch_filenames)
    
    return np.vstack(features), np.array(labels), filenames

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
    import numpy as np
    
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
    
    if 'torch' in globals():
        torch.cuda.empty_cache()
    
    return {
        'mean_corresponding_similarity': mean_corresponding_similarity
    }

def representational_similarity_analysis(line_features, color_features, verbose=False, layer_name=None, model_name=None):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    
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
    
    # DIAGNOSTIC 1: Check if corresponding features are identical (would indicate bug)
    identical_pairs = 0
    min_diff = float('inf')
    max_diff = 0
    mean_diff = 0
    
    for i in range(n_samples):
        diff = np.linalg.norm(line_features[i] - color_features[i])
        mean_diff += diff
        if diff < min_diff:
            min_diff = diff
        if diff > max_diff:
            max_diff = diff
        if diff < 1e-10:  # Essentially identical
            identical_pairs += 1
    
    mean_diff /= n_samples
    
    # DIAGNOSTIC 2: Check feature statistics
    line_feat_mean = np.mean(line_features)
    line_feat_std = np.std(line_features)
    color_feat_mean = np.mean(color_features)
    color_feat_std = np.std(color_features)
    
    # DIAGNOSTIC 3: Compute cosine similarity between corresponding features
    corresponding_cos_sims = []
    for i in range(n_samples):
        line_norm = line_features[i] / (np.linalg.norm(line_features[i]) + 1e-10)
        color_norm = color_features[i] / (np.linalg.norm(color_features[i]) + 1e-10)
        cos_sim = np.dot(line_norm, color_norm)
        corresponding_cos_sims.append(cos_sim)
    mean_corresponding_cos_sim = np.mean(corresponding_cos_sims)
    
    # Standard RSA: Compute full RDM (Representational Dissimilarity Matrix) then extract upper triangular
    # RDM is n x n matrix where RDM[i,j] = distance between samples i and j
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
    
    # DIAGNOSTIC: Verify pair correspondence for layer3 or verbose mode
    pair_verifications = []
    if verbose or layer_name == 'layer3':
        # Verify a few pairs manually to ensure RDM computation is correct
        check_pairs = [(0, 1), (0, 2), (1, 2)] if n_samples >= 3 else []
        for i, j in check_pairs:
            line_manual = np.linalg.norm(line_features[i] - line_features[j])
            color_manual = np.linalg.norm(color_features[i] - color_features[j])
            line_from_rdm = line_rdm[i, j]
            color_from_rdm = color_rdm[i, j]
            
            # Find index in upper triangular array
            idx_in_triu = np.where((triu_indices[0] == i) & (triu_indices[1] == j))[0]
            if len(idx_in_triu) > 0:
                all_match = (np.abs(line_manual - line_from_rdm) < 1e-6 and
                            np.abs(color_manual - color_from_rdm) < 1e-6 and
                            np.abs(line_from_rdm - line_distances[idx_in_triu[0]]) < 1e-6 and
                            np.abs(color_from_rdm - color_distances[idx_in_triu[0]]) < 1e-6)
                pair_verifications.append({
                    'pair': (i, j),
                    'line_manual': line_manual,
                    'line_rdm': line_from_rdm,
                    'line_triu': line_distances[idx_in_triu[0]],
                    'color_manual': color_manual,
                    'color_rdm': color_from_rdm,
                    'color_triu': color_distances[idx_in_triu[0]],
                    'all_match': all_match
                })
    
    # DIAGNOSTIC 4: Distance statistics
    line_dist_mean = np.mean(line_distances)
    line_dist_std = np.std(line_distances)
    line_dist_min = np.min(line_distances)
    line_dist_max = np.max(line_distances)
    
    color_dist_mean = np.mean(color_distances)
    color_dist_std = np.std(color_distances)
    color_dist_min = np.min(color_distances)
    color_dist_max = np.max(color_distances)
    
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
    # If there are ties or outliers, Spearman might be more robust, but Pearson is standard for RSA
    try:
        correlation, p_value = pearsonr(line_distances, color_distances)
    except ValueError:
        # Fallback to Spearman if Pearson fails (e.g., due to constant values)
        correlation, p_value = spearmanr(line_distances, color_distances)
    
    # Check for invalid correlation values
    if np.isnan(correlation) or np.isinf(correlation):
        # Try Spearman as fallback
        correlation, p_value = spearmanr(line_distances, color_distances)
        if np.isnan(correlation) or np.isinf(correlation):
            raise ValueError(f"Invalid correlation computed: {correlation}")
    
    # Print diagnostics if verbose or if suspicious values detected
    if verbose or (layer_name == 'layer3' and correlation > 0.6) or identical_pairs > 0:
        print(f"\n  [RSA DIAGNOSTICS] {model_name} - {layer_name}")
        print(f"    Feature shapes: line={line_features.shape}, color={color_features.shape}")
        print(f"    Corresponding features check:")
        print(f"      - Identical pairs: {identical_pairs}/{n_samples}")
        print(f"      - L2 diff: min={min_diff:.6f}, max={max_diff:.6f}, mean={mean_diff:.6f}")
        print(f"      - Mean cosine sim (corresponding): {mean_corresponding_cos_sim:.4f}")
        print(f"    Feature statistics:")
        print(f"      - Line: mean={line_feat_mean:.6f}, std={line_feat_std:.6f}")
        print(f"      - Color: mean={color_feat_mean:.6f}, std={color_feat_std:.6f}")
        print(f"    Distance statistics:")
        print(f"      - Line distances: mean={line_dist_mean:.6f}, std={line_dist_std:.6f}, "
              f"min={line_dist_min:.6f}, max={line_dist_max:.6f}")
        print(f"      - Color distances: mean={color_dist_mean:.6f}, std={color_dist_std:.6f}, "
              f"min={color_dist_min:.6f}, max={color_dist_max:.6f}")
        print(f"      - Distance variances: line_var={line_var:.6f}, color_var={color_var:.6f}")
        if pair_verifications:
            print(f"    Pair verification (RDM correctness check):")
            for pv in pair_verifications:
                status = "✓" if pv['all_match'] else "✗"
                print(f"      {status} Pair {pv['pair']}: line={pv['line_triu']:.4f}, color={pv['color_triu']:.4f}")
        print(f"    RSA result: correlation={correlation:.4f}, p_value={p_value:.6f}")
    
    return {
        'rsa_correlation': correlation,
        'rsa_p_value': p_value
    }

def procrustes_analysis(line_features, color_features, layer_name=None):
    """
    Memory-efficient Procrustes analysis using PCA dimensionality reduction.
    Reduces features to manageable dimensions before applying Procrustes.
    """
    import numpy as np
    from scipy.linalg import orthogonal_procrustes
    from sklearn.decomposition import PCA
    
    if isinstance(line_features, torch.Tensor):
        line_features = line_features.numpy()
    if isinstance(color_features, torch.Tensor):
        color_features = color_features.numpy()
    
    # Determine target dimensionality (use smaller of: n_samples-1, 50, or original dims)
    n_samples = min(len(line_features), len(color_features))
    target_dims = min(n_samples - 1, 50, line_features.shape[1], color_features.shape[1])
    
    # Apply PCA to reduce dimensionality for memory efficiency
    pca = PCA(n_components=target_dims)
    
    # Fit PCA on combined data to ensure same transformation
    combined_features = np.vstack([line_features, color_features])
    pca.fit(combined_features)
    
    # Transform both feature sets
    line_reduced = pca.transform(line_features)
    color_reduced = pca.transform(color_features)
    
    # Center the reduced feature matrices
    line_centered = line_reduced - line_reduced.mean(axis=0)
    color_centered = color_reduced - color_reduced.mean(axis=0)
    
    # Find optimal rotation matrix using Procrustes analysis
    R, scale = orthogonal_procrustes(line_centered, color_centered)
    
    # Calculate rotation angle from rotation matrix
    trace_R = np.trace(R)
    n_dims = R.shape[0]
    
    # Handle different dimensionality cases
    if n_dims == 2:
        cos_angle = trace_R / 2.0
    else:
        cos_angle = (trace_R - 1) / (n_dims - 1)
    
    # Ensure valid range for arccos
    cos_angle = np.clip(cos_angle, -1, 1)
    rotation_angle_rad = np.arccos(cos_angle)
    rotation_angle_deg = np.degrees(rotation_angle_rad)
    
    # Calculate alignment quality
    line_aligned = line_centered @ R * scale
    alignment_error = np.mean(np.linalg.norm(line_aligned - color_centered, axis=1))
    
    # Clean up memory
    del line_reduced, color_reduced, line_centered, color_centered, combined_features
    if 'torch' in globals():
        torch.cuda.empty_cache()
    
    return {
        'rotation_angle': rotation_angle_deg,
        'scale_factor': scale,
        'alignment_error': alignment_error,
        'pca_components': target_dims,
        'explained_variance_ratio': pca.explained_variance_ratio_.sum()
    }

def analyze_structural_vs_color_prior(line_features, color_features):
    import numpy as np
    from sklearn.decomposition import PCA
    
    def cosine_similarity_pair(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        return np.dot(a_norm, b_norm)
    
    diagonal_similarities = []
    for i in range(len(line_features)):
        sim = cosine_similarity_pair(line_features[i], color_features[i])
        diagonal_similarities.append(sim)
    
    diagonal_similarities = np.array(diagonal_similarities)
    
    line_variance = np.var(line_features, axis=0).mean()
    color_variance = np.var(color_features, axis=0).mean()
    
    line_pca = PCA(n_components=min(10, line_features.shape[1]))
    color_pca = PCA(n_components=min(10, color_features.shape[1]))
    
    line_pca.fit(line_features)
    color_pca.fit(color_features)
    
    line_explained_var = line_pca.explained_variance_ratio_.sum()
    color_explained_var = color_pca.explained_variance_ratio_.sum()
    
    return {
        'mean_cosine_similarity': diagonal_similarities.mean(),
        'line_variance': line_variance,
        'color_variance': color_variance,
        'variance_ratio': line_variance / color_variance,
        'line_pca_explained_var': line_explained_var,
        'color_pca_explained_var': color_explained_var,
        'similarity_std': diagonal_similarities.std()
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_paths = {
        'Color Only': '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_color.pth',
        'Line Only': '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_line.pth',
        'Line → Color': '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_linecolor.pth',
        'Color → Line': '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_colorline.pth',
        'Interleaved': '/user_data/georgeli/workspace/STL10-Resnet-18/Models/resnet18_interleaved.pth'
    }
    
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    
    transform = get_transform()
    
    line_dataset = STL10Dataset(
        img_dir='/user_data/georgeli/workspace/STL10-Resnet-18/STL10-Line/test_images',
        json_file='/user_data/georgeli/workspace/STL10-Resnet-18/STL10-Line/test.json',
        transform=transform
    )
    
    color_dataset = STL10Dataset(
        img_dir='/user_data/georgeli/workspace/STL10-Resnet-18/STL10/test_images',
        json_file='/user_data/georgeli/workspace/STL10-Resnet-18/STL10/test.json',
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
                rsa_results = representational_similarity_analysis(
                    line_features, color_features, 
                    verbose=False, layer_name=layer_name, model_name=model_name
                )
                
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
    
    output_dir = '/user_data/georgeli/workspace/STL10-Resnet-18/manifold_alignment/results'
    
    save_results(results, output_dir)
    
    print("\nCreating visualizations...")
    create_visualizations(results, output_dir)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")
    
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
