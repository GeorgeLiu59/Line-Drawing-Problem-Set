# Computational Perception HW5 Code & Results

Code for HW5 of 15-387 for Fall 2025.

## Overview

Analysis of ResNet-18 models trained on color photographs vs. line drawings to understand how transfer learning affects representational manifolds. We evaluate shape bias using the cue-conflict dataset to compare models' reliance on shape versus texture cues.

## Requirements

**Python:** 3.8+

Install dependencies:

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn pillow tqdm pandas click
```

### Model-vs-Human Setup

The `model-vs-human/` directory contains a modified version of the [model-vs-human](https://github.com/bethgelab/model-vs-human) toolbox for shape-bias evaluation.

**Note:** The `evaluate.py` script automatically handles path setup, so installation is optional. However, if you want to use the package from other scripts, you can install it.

**Installation (optional):**

1. Set the environment variable (from the `model-vs-human/` directory):
```bash
cd model-vs-human
export MODELVSHUMANDIR=$(pwd)
```

2. Install the package in editable mode:
```bash
pip install -e .
```

## Project Structure

```
CP-PSET/
├── README.md
├── rsa_analysis.py           # Representation alignment analysis (cosine similarity, RSA)
├── correlation_matrices.py  # Correlation matrix comparison and visualization
├── accuracy.py               # Model accuracy evaluation
├── Models/                  # Trained ResNet-18 checkpoints
│   ├── resnet18_color.pth
│   ├── resnet18_line.pth
│   ├── resnet18_linecolor.pth
│   ├── resnet18_colorline.pth
│   └── resnet18_interleaved.pth
├── Results/                 # Analysis outputs
│   ├── correlation_matrices_results/
│   │   ├── domain_comparison_matrices_layer1.png
│   │   ├── domain_comparison_matrices_layer2.png
│   │   ├── domain_comparison_matrices_layer3.png
│   │   └── domain_comparison_matrices_layer4.png
│   ├── rsa_analysis_results.out
│   └── shape_bias_matrix.pdf
└── model-vs-human/          # Shape-bias evaluation toolbox
    ├── examples/
    │   └── evaluate.py      # Main evaluation script
    └── figures/              # Generated shape-bias plots
        └── shape-bias-figures/
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

**Output:** Results printed to console and saved to output log.

### `correlation_matrices.py`
Compares models using correlation matrices to visualize same-domain vs cross-domain patterns across ResNet-18 residual blocks (layer1-4). Uses 100 images per class.

```bash
python correlation_matrices.py
```

**Output:**
- `domain_comparison_matrices_layer{1-4}.png`: 2x5 grid visualizations showing same-domain (top row) vs cross-domain (bottom row) correlation matrices for all 5 models at each layer

### `model-vs-human/examples/evaluate.py`
Evaluates ResNet-18 models on the cue-conflict dataset and generates shape-bias visualizations. This script:
1. Evaluates models (`resnet18_base` and `resnet18_linecolor`) on the cue-conflict dataset
2. Generates shape-bias plots (matrixplot and boxplot)

**Usage:**
```bash
cd model-vs-human/examples
python evaluate.py
```

**Output:** Results are saved to `model-vs-human/figures/shape-bias-figures/`:
- `cue-conflict_shape-bias_matrixplot.pdf`: Shape-bias matrix visualization
- `cue-conflict_shape-bias_boxplot.pdf`: Shape-bias boxplot comparison

**Note:** The script automatically downloads the cue-conflict dataset on first run if not already present.

## Results

Analysis outputs are saved in the `Results/` directory:

- **`correlation_matrices_results/`**: Contains correlation matrix visualizations (one PNG per layer)
- **`rsa_analysis_results.out`**: Output log from RSA analysis script
- **`shape_bias_matrix.pdf`**: Shape-bias matrix visualization

Shape-bias evaluation results are also saved in `model-vs-human/figures/shape-bias-figures/`:
- **`cue-conflict_shape-bias_matrixplot.pdf`**: Matrix visualization comparing shape vs texture bias
- **`cue-conflict_shape-bias_boxplot.pdf`**: Boxplot comparing shape bias across models

## Data Requirements

Update paths in scripts to point to your STL-10 datasets:
- Color images: directory with `test.json` and `test_images/`
- Line drawings: directory with `test.json` and `test_images/`

