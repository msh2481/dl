"""
Learning rate finder implementation.
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def find_lr(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_iter: int = 200,
    output_path: str = "lr_finder_plot.png",
) -> Tuple[List[float], List[float]]:
    """
    Run learning rate finder to determine optimal learning rate.

    Args:
        model: The model to train
        train_loader: Training dataloader
        optimizer: Optimizer (will be modified)
        criterion: Loss function
        device: Device to run on
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iter: Number of iterations to run
        output_path: Path to save the plot

    Returns:
        Tuple of (learning_rates, losses)
    """
    model.train()

    # Calculate multiplication factor for exponential increase
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)

    # Set initial learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = start_lr

    learning_rates = []
    losses = []
    best_loss = float("inf")
    batch_iter = iter(train_loader)

    pbar = tqdm(range(num_iter), desc="LR Finder")

    for iteration in pbar:
        # Get next batch
        try:
            inputs, targets = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            inputs, targets = next(batch_iter)

        inputs, targets = inputs.to(device), targets.to(device)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Track loss
        losses.append(loss.item())

        # Check if loss is exploding
        if loss.item() > best_loss * 4:
            logger.warning(f"Stopping early at iteration {iteration}: loss is exploding")
            break

        best_loss = min(best_loss, loss.item())

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_mult

        pbar.set_postfix({"lr": f"{current_lr:.2e}", "loss": f"{loss.item():.4f}"})

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True, alpha=0.3)

    # Try to suggest an LR (where gradient is steepest)
    if len(losses) > 10:
        # Smooth losses
        smoothed_losses = np.convolve(losses, np.ones(5) / 5, mode="valid")
        # Find steepest gradient
        gradients = np.gradient(smoothed_losses)
        min_gradient_idx = np.argmin(gradients)
        suggested_lr = learning_rates[min_gradient_idx]

        plt.axvline(suggested_lr, color="red", linestyle="--", alpha=0.7)
        plt.text(
            suggested_lr,
            max(losses) * 0.9,
            f"Suggested LR: {suggested_lr:.2e}",
            color="red",
        )

    # Save plot
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"LR finder plot saved to {output_path}")

    return learning_rates, losses
