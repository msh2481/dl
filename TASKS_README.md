# Audio Deep Learning Tasks

This directory contains 5 standalone Python scripts for audio classification tasks using AudioMNIST dataset.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Task Scripts

Each task is a standalone script that produces outputs in its own directory:

### Task 1: Supervised Learning (`task1.py`)
Trains 1D and 2D supervised models from scratch.

**Output:** `task1_supervised/`
- `1d_checkpoints/best.ckpt` - Best 1D model checkpoint
- `2d_checkpoints/best.ckpt` - Best 2D model checkpoint
- `metrics.json` - Test/val accuracies in JSON format
- `metrics.txt` - Human-readable metrics

### Task 2: Contrastive Learning (`task2.py`)
Trains multi-format contrastive model (InfoNCE loss) and evaluates with linear probes.

**Output:** `task2_contrastive/`
- `checkpoints/best.ckpt` - Best contrastive model checkpoint
- `metrics.json` - Linear probe accuracies (1D, 2D, concat)
- `metrics.txt` - Human-readable metrics

### Task 3: t-SNE Visualization (`task3.py`)
Generates t-SNE plots for embeddings from supervised and contrastive models.

**Output:** `task3_visualization/`
- 8 PNG files: `{model}_{coloring}.png`
  - Models: `supervised_1d`, `supervised_2d`, `contrastive_1d`, `contrastive_2d`
  - Colorings: `digit`, `speaker`
- `metrics.txt` - List of generated plots

### Task 4: Voice Biometrics (`task4.py`)
Speaker identification using k-NN on contrastive embeddings.

**Output:** `task4_biometrics/`
- `metrics.json` - k-NN accuracies (1D, 2D, concat)
- `metrics.txt` - Human-readable metrics

### Task 5: Fine-tuning (`task5.py`)
Fine-tunes pre-trained contrastive encoders for digit classification.

**Output:** `task5_finetune/`
- 6 checkpoint directories: `{encoder}_{frozen/finetuned}_checkpoints/best.ckpt`
  - Encoders: `1d`, `2d`, `concat`
  - Modes: `frozen` (linear probe), `finetuned` (end-to-end)
- `metrics.json` - Test/val accuracies for all 6 configs
- `metrics.txt` - Human-readable metrics

## Common Arguments

All scripts support:

- `--ratio FLOAT` - Dataset fraction to use (default: 1.0). Use 0.01 for smoke tests.
- `--epochs INT` - Number of training epochs (default varies by task)
- `--batch-size INT` - Batch size (default: 64)
- `--lr FLOAT` - Learning rate (default varies by task)
- `--num-workers INT` - DataLoader workers (default: 4)

Tasks 3, 4, 5 also support:
- `--task2-checkpoint PATH` - Path to contrastive model checkpoint

Task 3 additionally supports:
- `--task1-checkpoint PATH` - Path to task1 output directory

## Usage Examples

### Full training (all data)
```bash
python task1.py --epochs 50
python task2.py --epochs 100
python task3.py
python task4.py
python task5.py --epochs 50
```

### Smoke test (1% data, 1 epoch)
```bash
python task1.py --ratio 0.01 --epochs 1 --num-workers 0
python task2.py --ratio 0.01 --epochs 1 --num-workers 0
python task3.py --ratio 0.01 --num-workers 0
python task4.py --ratio 0.01 --num-workers 0
python task5.py --ratio 0.01 --epochs 1 --num-workers 0
```

### Run all tasks sequentially
```bash
./run_all.sh 1.0 50    # ratio=1.0, epochs=50
./run_all.sh 0.01 1    # ratio=0.01, epochs=1 (smoke test)
```

## Model Architecture

All models use smaller, faster architectures optimized for Mac training:
- **1D Encoder**: 2-layer CNN (32→64 channels), 128-dim embeddings
- **2D Encoder**: 2-layer CNN (32→64 channels), 128-dim embeddings
- **Total parameters**: ~69K (62x smaller than original design)
- **Speedup**: 10-20x faster training

## Notes

- All tasks use PyTorch Lightning for training
- Checkpoints are saved based on validation metrics
- Early stopping is enabled (patience=10 epochs)
- DataLoaders use `persistent_workers=True` for efficiency
- Task 3 requires completed Task 1 and Task 2
- Tasks 4 and 5 require completed Task 2
