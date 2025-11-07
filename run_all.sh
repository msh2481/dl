#!/bin/bash
set -e

# Optional argument: epoch ratio (0.0 to 1.0)
EPOCH_RATIO=${1:-1.0}

echo "========================================"
echo "Starting full training and evaluation pipeline"
echo "Epoch ratio: $EPOCH_RATIO"
echo "========================================"

# Step 0: Supervised baseline
echo ""
echo "Step 0: Training supervised baseline..."
python train_stl10.py --epoch-ratio $EPOCH_RATIO
mv checkpoints/best-*.ckpt checkpoints/supervised.ckpt

# Step 1: SimCLR
echo ""
echo "Step 1: Training SimCLR..."
python train_simclr.py --epoch-ratio $EPOCH_RATIO

# Step 2: BYOL
echo ""
echo "Step 2: Training BYOL..."
python train_byol.py --epoch-ratio $EPOCH_RATIO

# Step 3: t-SNE visualization before fine-tuning
echo ""
echo "Step 3: Generating t-SNE visualizations (before fine-tuning)..."
python tsne_viz.py --checkpoint checkpoints/supervised.ckpt --output tsne_supervised.png
python tsne_viz.py --checkpoint checkpoints/simclr-best-*.ckpt --output tsne_simclr.png
python tsne_viz.py --checkpoint checkpoints/byol-best-*.ckpt --output tsne_byol.png

# Step 5: Fine-tuning
echo ""
echo "Step 5: Fine-tuning SimCLR..."
python train_stl10.py --checkpoint checkpoints/simclr-best-*.ckpt --epochs 100 --lr 1e-4 --epoch-ratio $EPOCH_RATIO
mv checkpoints/best-*.ckpt checkpoints/simclr-finetuned.ckpt

echo ""
echo "Step 5: Fine-tuning BYOL..."
python train_stl10.py --checkpoint checkpoints/byol-best-*.ckpt --epochs 100 --lr 1e-4 --epoch-ratio $EPOCH_RATIO
mv checkpoints/best-*.ckpt checkpoints/byol-finetuned.ckpt

# Step 3 (continued): t-SNE visualization after fine-tuning
echo ""
echo "Step 3: Generating t-SNE visualizations (after fine-tuning)..."
python tsne_viz.py --checkpoint checkpoints/simclr-finetuned.ckpt --output tsne_simclr_finetuned.png
python tsne_viz.py --checkpoint checkpoints/byol-finetuned.ckpt --output tsne_byol_finetuned.png

# Step 6: OOD evaluation
echo ""
echo "Step 6: OOD evaluation on CIFAR-10..."
echo "Supervised:"
python ood_eval.py --checkpoint checkpoints/supervised.ckpt

echo ""
echo "SimCLR (linear probe):"
python ood_eval.py --checkpoint checkpoints/simclr-best-*.ckpt

echo ""
echo "SimCLR (fine-tuned):"
python ood_eval.py --checkpoint checkpoints/simclr-finetuned.ckpt

echo ""
echo "BYOL (linear probe):"
python ood_eval.py --checkpoint checkpoints/byol-best-*.ckpt

echo ""
echo "BYOL (fine-tuned):"
python ood_eval.py --checkpoint checkpoints/byol-finetuned.ckpt

echo ""
echo "========================================"
echo "Pipeline completed!"
echo "========================================"
