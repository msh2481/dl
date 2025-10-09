# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning project for image classification using ResNet-18 on animal datasets. The project includes supervised training with optional pretraining, learning rate finding, and early stopping.

## Development Setup

```bash
uv sync
source .venv/bin/activate
```

## Common Commands

### Training

```bash
# Create baseline pretrained model
python pretrain.py

# Supervised training with pretrained checkpoint
python supervised.py checkpoints/pretrained_baseline.pt 1e-3

# Train from scratch
python supervised.py none 1e-3

# Find optimal learning rate
python supervised.py checkpoints/pretrained_baseline.pt find
```

The supervised training script accepts these key options:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--weight-decay`: Weight decay for AdamW (default: 0.01)
- `--early-stopping-patience`: Patience in epochs (default: 10)
- `--save-every-n-steps`: Checkpoint frequency (default: 100)

### W&B Integration

Training automatically uses Weights & Biases if `WANDB_API_KEY` environment variable is set. Otherwise, only local logging is used.

## Architecture

### Data Pipeline (`utils/data.py`)

- Uses ImageFolder format: `data/train/labeled/[class_name]/[images]`
- Train/val split is stratified by default (80/20)
- Training transforms include: resize, random crop, horizontal flip, color jitter
- Validation transforms: resize + center crop only
- Uses ImageNet normalization statistics

### Training Loop (`utils/training.py`)

The `Trainer` class handles:
- Training/validation loops with progress bars
- Automatic checkpointing (best model + periodic step checkpoints)
- Early stopping based on validation accuracy
- Optional W&B logging
- Cosine annealing LR scheduler support

Key features:
- Best checkpoint saved based on validation accuracy
- Step checkpoints saved every N steps to `checkpoints/step_{global_step}.pt`
- Final checkpoint saved at end as `checkpoints/final.pt`

### Learning Rate Finder (`utils/lr_finder.py`)

Implements the LR range test:
- Exponentially increases LR from start_lr to end_lr
- Tracks loss at each LR
- Stops early if loss explodes (>4x best loss)
- Suggests optimal LR based on steepest gradient
- Saves plot to `lr_finder_plot.png`

## Model Details

- Architecture: ResNet-18 (torchvision.models)
- Output: 10 classes (animal categories)
- Optimizer: AdamW with configurable weight decay
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss

## Directory Structure

```
data/
  train/
    labeled/     # Supervised training data (ImageFolder format)
    unlabeled/   # Unlabeled data (not currently used)
  test/          # Test images
checkpoints/     # Model checkpoints
utils/           # Training utilities
results/         # Evaluation metrics (JSON files)
fronts/          # Pareto front plots
```

## GPU Environment Notes

If encountering torch errors on Constructor GPU, use:
```bash
export PATH="/home/coder/project/dl/.venv/bin:$PATH"
hash -r
unset PYTHONPATH
```
