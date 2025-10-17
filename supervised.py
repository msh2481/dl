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
from sklearn.linear_model import LogisticRegression
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision import models
from tqdm import tqdm

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


@torch.no_grad()
def extract_features_and_labels(model: nn.Module, dataloader, device: torch.device):
    """Extract features from model without final FC layer."""
    model.eval()

    # Temporarily replace FC layer with identity
    original_fc = model.fc
    model.fc = nn.Identity()

    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu())
        all_labels.append(labels)

    # Restore original FC layer
    model.fc = original_fc

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


def initialize_head_with_logreg(
    model: nn.Module, train_loader, device: torch.device
) -> nn.Module:
    """
    Initialize the final FC layer using LogisticRegression on extracted features.

    This provides a good initialization for fine-tuning from unsupervised checkpoints.
    """
    # Extract features from training data
    train_features, train_labels = extract_features_and_labels(
        model, train_loader, device
    )

    # Fit LogisticRegression
    logger.info("Fitting LogisticRegression on extracted features...")
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)
    clf.fit(train_features, train_labels)

    # Copy weights to model's FC layer
    # LogisticRegression: y = X @ coef_.T + intercept_
    # PyTorch Linear: y = X @ weight.T + bias
    # So we can directly copy: weight = coef_, bias = intercept_
    with torch.no_grad():
        model.fc.weight.copy_(torch.from_numpy(clf.coef_).float())
        model.fc.bias.copy_(torch.from_numpy(clf.intercept_).float())

    logger.info("Classification head initialized with LogisticRegression weights")

    return model


@app.command()
def main(
    checkpoint: str = typer.Argument(
        ..., help="Path to pretrained checkpoint or 'none' for from-scratch baseline"
    ),
    lr: str = typer.Argument(..., help="Learning rate (float) or 'find' for LR finder"),
    weight_decay: float = typer.Option(0.003, help="Weight decay for AdamW"),
    beta2: float = typer.Option(0.99, help="Beta2 for AdamW optimizer"),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    batch_size: int = typer.Option(8, help="Batch size"),
    save_every_n_steps: int = typer.Option(100, help="Save checkpoint every N steps"),
    early_stopping_patience: int = typer.Option(
        3000, help="Early stopping patience (epochs)"
    ),
    warmup_epochs: int = typer.Option(5, help="Number of warmup epochs"),
    ema_span: int = typer.Option(10, help="EMA span in epochs"),
    use_augmentation: bool = typer.Option(
        True, help="Use data augmentation on training set"
    ),
    augmentation_strength: str = typer.Option(
        "medium", help="Augmentation strength: 'light', 'medium', or 'strong'"
    ),
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
    logger.info(f"Data augmentation: {'enabled' if use_augmentation else 'disabled'} (strength: {augmentation_strength})")
    train_loader, val_loader, class_names = get_dataloaders(
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed,
        use_augmentation=use_augmentation,
        augmentation_strength=augmentation_strength,
    )
    logger.info(f"Classes: {class_names}")

    # Load model
    model = load_model(checkpoint, device)
    logger.info(
        f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Initialize head using LogisticRegression on extracted features
    if checkpoint != "none":
        logger.info("Initializing classification head with LogisticRegression...")
        model = initialize_head_with_logreg(model, train_loader, device)

    # Setup optimizer and criterion
    criterion = nn.CrossEntropyLoss()

    # LR Finder mode
    if lr == "find":
        logger.info("Running LR Finder...")
        # Use a temporary optimizer for LR finding
        temp_optimizer = AdamW(
            model.parameters(), lr=1e-10, weight_decay=weight_decay, betas=(0.9, beta2)
        )

        find_lr(
            model=model,
            train_loader=train_loader,
            optimizer=temp_optimizer,
            criterion=criterion,
            device=device,
            start_lr=1e-10,
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
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, beta2),
    )

    # Setup scheduler: warmup + cosine annealing
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=0.1 * learning_rate
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Initialize wandb
    use_wandb = WANDB_AVAILABLE and os.getenv("WANDB_API_KEY") is not None
    if use_wandb:
        logger.info("Initializing W&B logging...")
        wandb.init(
            project="animal-classification",
            config={
                "checkpoint": checkpoint,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "beta2": beta2,
                "epochs": epochs,
                "batch_size": batch_size,
                "val_split": val_split,
                "seed": seed,
                "architecture": "resnet18",
                "warmup_epochs": warmup_epochs,
                "ema_span": ema_span,
                "use_augmentation": use_augmentation,
                "augmentation_strength": augmentation_strength,
            },
        )
        logger.info(f"W&B run: {wandb.run.name} ({wandb.run.url})")
    else:
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available - wandb package not installed")
        else:
            logger.warning("W&B disabled - WANDB_API_KEY not set")

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
        ema_span=ema_span,
    )
    trainer.set_scheduler(scheduler)

    # Train
    trainer.train(num_epochs=epochs)

    # Cleanup
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    app()
