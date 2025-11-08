# Computational Perception 15-387 HW5

Code for HW5 of 15-387 for Fall 2025.

## Overview

Analysis of ResNet-18 models trained on color photographs vs. line drawings to understand how transfer learning affects representational manifolds.

## Requirements

Install dependencies:

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn pandas pillow tqdm
```

## Project Structure

```
Line-Drawing-Problem-Set/
├── README.md
├── rsa_analysis.py           # Representation alignment analysis (cosine similarity, RSA)
├── correlation_matrices.py  # Correlation matrix comparison and visualization
├── accuracy.py               # Model accuracy evaluation
└── Models/                  # Trained ResNet-18 checkpoints
    ├── resnet18_color.pth
    ├── resnet18_line.pth
    ├── resnet18_linecolor.pth
    ├── resnet18_colorline.pth
    └── resnet18_interleaved.pth
```

## Models

Five ResNet-18 models with different training strategies:
- **Color Only** (`resnet18_color.pth`): Trained on color images only
- **Line Only** (`resnet18_line.pth`): Trained on line drawings only
- **Line→Color** (`resnet18_linecolor.pth`): Transfer learning from line to color
- **Color→Line** (`resnet18_colorline.pth`): Transfer learning from color to line
- **Interleaved** (`resnet18_interleaved.pth`): Combined training on both modalities

## Scripts

### `accuracy.py`
Evaluates model accuracy on test sets. Update `MODEL_PATHS`, `TEST_JSON_PATH`, and `TEST_IMAGES_DIR` in the script.

```bash
python accuracy.py
```

### `rsa_analysis.py`
Performs representation alignment analysis (cosine similarity, RSA) across ResNet-18 residual blocks.

**Layers analyzed:**
- **layer1**: First residual block (2 basic blocks, 64 channels)
- **layer2**: Second residual block (2 basic blocks, 128 channels)
- **layer3**: Third residual block (2 basic blocks, 256 channels)
- **layer4**: Fourth residual block (2 basic blocks, 512 channels)

These layers are extracted after the initial conv1, bn1, relu, and maxpool layers, representing progressively deeper feature representations.

```bash
python rsa_analysis.py
```

**Output:** JSON results, CSV summary, and visualization plots.

### `correlation_matrices.py`
Compares models using correlation matrices to visualize same-domain vs cross-domain patterns across ResNet-18 residual blocks (layer1-4). Uses 100 images per class.

```bash
python correlation_matrices.py
```

**Output:** Correlation matrix visualizations (same-domain vs cross-domain for all 5 models).

## Data Requirements

Update paths in scripts to point to your STL-10 datasets:
- Color images: directory with `test.json` and `test_images/`
- Line drawings: directory with `test.json` and `test_images/`

