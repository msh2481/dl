import torch
import typer
from loguru import logger
from torch import nn
from tqdm import tqdm

from data import get_dataloaders, get_datasets
from random_texts import CLIPZeroShotClassifier


def save_model(model: torch.nn.Module, filepath: str) -> None:
    torch.save(model.state_dict(), filepath)
    logger.info(f"Model saved to {filepath}")


def main(
    fraction: float = typer.Option(1e-3, help="Fraction of dataset to use"),
    batch_size: int = typer.Option(256, help="Batch size for training"),
    lr: float = typer.Option(1e-5, help="Learning rate"),
    weight_decay: float = typer.Option(0.1, help="Weight decay for optimizer"),
    warmup_fraction: float = typer.Option(
        0.1, help="Fraction of total steps for warmup"
    ),
    max_grad_norm: float = typer.Option(1.0, help="Maximum gradient norm for clipping"),
) -> None:
    logger.info(
        f"Starting fine-tuning with fraction: {fraction}, batch size: {batch_size}, lr: {lr}"
    )

    datasets, classnames = get_datasets(fraction=fraction)
    for name, dataset in datasets.items():
        logger.info(
            f"{name}: {len(dataset)} samples -> {(len(dataset) + batch_size - 1) // batch_size} batches"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ft_model = CLIPZeroShotClassifier(classnames)
    logger.info(f"Model dtype: {ft_model.dtype}, device: {ft_model.device}")
    dataloaders = get_dataloaders(datasets, ft_model.preprocess, batch_size=batch_size)
    logger.info("Dataloaders created")

    optimizer = torch.optim.AdamW(
        ft_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = len(dataloaders["ID"])
    warmup_steps = int(warmup_fraction * total_steps)
    logger.info(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    pbar = tqdm(dataloaders["ID"], desc="Fine-tuning")
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Check for NaN/inf in inputs
        if torch.isnan(images).any() or torch.isinf(images).any():
            logger.error(f"NaN/Inf detected in images at step {step}")
            continue
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            logger.error(f"NaN/Inf detected in labels at step {step}")
            continue

        logits = ft_model(images)

        # Check for NaN/inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error(f"NaN/Inf detected in logits at step {step}")
            logger.info(
                f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}"
            )
            continue

        loss = nn.functional.cross_entropy(logits, labels)

        # Check for NaN/inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN/Inf loss detected at step {step}")
            logger.info(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}")
            logger.info(f"Labels: {labels}")
            continue

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % 10 == 0:
            logger.info(f"Step {step}, Loss: {loss.item():.4f}")
            save_model(ft_model, "ft_model.pth")

        optimizer.zero_grad()
        loss.backward()

        # Check gradients before clipping
        total_norm = torch.nn.utils.clip_grad_norm_(
            ft_model.parameters(), max_grad_norm
        )
        if torch.isnan(total_norm) or torch.isinf(total_norm):
            logger.error(f"NaN/Inf gradient norm detected at step {step}: {total_norm}")
            continue

        if step % 1 == 0:
            logger.info(f"Gradient norm: {total_norm:.4f}")

        optimizer.step()
        scheduler.step()

    save_model(ft_model, "ft_model.pth")
    logger.success("Fine-tuning completed")


if __name__ == "__main__":
    typer.run(main)
