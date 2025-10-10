"""
Supervised training script for ResNet-18 on labeled animal data.

Usage:
    # Train with pretrained checkpoint
    uv run supervised.py checkpoints/pretrained_baseline.pt 1e-3

    # Train from scratch
    uv run supervised.py none 1e-3

    # Find optimal learning rate
    uv run supervised.py checkpoints/pretrained_baseline.pt find
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import typer
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import models

from utils.data import get_dataloaders
from utils.lr_finder import find_lr
from utils.training import Trainer

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


app = typer.Typer()


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load ResNet-18 model with optional pretrained weights.

    Args:
        checkpoint_path: Path to checkpoint or "none" for random initialization
        device: Device to load model on

    Returns:
        ResNet-18 model
    """
    model = models.resnet18(num_classes=10)

    if checkpoint_path != "none":
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model = model.to(device)
    return model


@app.command()
def main(
    checkpoint: str = typer.Argument(
        ..., help="Path to pretrained checkpoint or 'none' for from-scratch baseline"
    ),
    lr: str = typer.Argument(..., help="Learning rate (float) or 'find' for LR finder"),
    weight_decay: float = typer.Option(0.03, help="Weight decay for AdamW"),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    save_every_n_steps: int = typer.Option(100, help="Save checkpoint every N steps"),
    early_stopping_patience: int = typer.Option(
        30, help="Early stopping patience (epochs)"
    ),
    warmup_epochs: int = typer.Option(5, help="Number of warmup epochs"),
    val_split: float = typer.Option(0.2, help="Validation split ratio"),
    num_workers: int = typer.Option(4, help="Number of dataloader workers"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Train ResNet-18 with supervised learning on labeled animal data."""
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, class_names = get_dataloaders(
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed,
    )
    logger.info(f"Classes: {class_names}")

    # Load model
    model = load_model(checkpoint, device)
    logger.info(
        f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Setup optimizer and criterion
    criterion = nn.CrossEntropyLoss()

    # LR Finder mode
    if lr == "find":
        logger.info("Running LR Finder...")
        # Use a temporary optimizer for LR finding
        temp_optimizer = AdamW(model.parameters(), lr=1e-7, weight_decay=weight_decay)

        find_lr(
            model=model,
            train_loader=train_loader,
            optimizer=temp_optimizer,
            criterion=criterion,
            device=device,
            start_lr=1e-7,
            end_lr=1.0,
            num_iter=200,
            output_path="lr_finder_plot.png",
        )

        logger.info("LR finder complete. Check lr_finder_plot.png for results.")
        return

    # Convert lr to float
    try:
        learning_rate = float(lr)
    except ValueError:
        raise ValueError(f"Invalid learning rate: {lr}. Must be a float or 'find'")

    # Setup optimizer with actual learning rate
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Setup scheduler: warmup + cosine annealing
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and os.getenv("WANDB_API_KEY") is not None
    if use_wandb:
        wandb.init(
            project="animal-classification",
            config={
                "checkpoint": checkpoint,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "batch_size": batch_size,
                "val_split": val_split,
                "seed": seed,
                "architecture": "resnet18",
            },
        )

    # Setup trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_every_n_steps=save_every_n_steps,
        early_stopping_patience=early_stopping_patience,
        use_wandb=use_wandb,
    )
    trainer.set_scheduler(scheduler)

    # Train
    trainer.train(num_epochs=epochs)

    # Cleanup
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    app()
